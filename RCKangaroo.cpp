// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

// Statistics tracking
std::atomic<u64> gTameCount;
std::atomic<u64> gWild1Count;
std::atomic<u64> gWild2Count;
EcInt gLowestGap;
EcInt gEstimatedKey;
bool gHasLowestGap;
bool gHasEstimatedKey;

std::mutex gGapMutex;

u32 gDP;
u32 gRangeBits;
EcInt gStart;
EcInt gEnd;
EcInt gRangeWidth;
bool gStartSet;
bool gEndSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;

// Gap tracking helpers
struct DistanceEntry
{
        EcInt dist;
        int type;
};

struct DistanceEntryLess
{
        bool operator()(const DistanceEntry& a, const DistanceEntry& b) const
        {
                for (int i = 4; i >= 0; --i)
                {
                        if (a.dist.data[i] != b.dist.data[i])
                                return a.dist.data[i] < b.dist.data[i];
                }
                return a.type < b.type;
        }
};

std::multiset<DistanceEntry, DistanceEntryLess> gTameDistances;
std::multiset<DistanceEntry, DistanceEntryLess> gWild1Distances;
std::multiset<DistanceEntry, DistanceEntryLess> gWild2Distances;
DistanceEntry gBestDistanceA;
DistanceEntry gBestDistanceB;
bool gHasGapPair;

struct DpMeta
{
        EcInt dist;
        int type;
};

static constexpr size_t kDpMetaQueueSize = 1u << 20;
static_assert((kDpMetaQueueSize & (kDpMetaQueueSize - 1)) == 0, "kDpMetaQueueSize must be power of two");
static std::vector<DpMeta> gDpMetaQueue;
static std::atomic<size_t> gDpMetaHead{0};
static std::atomic<size_t> gDpMetaTail{0};
static std::atomic<bool> gDpMetaRunning{false};
static std::thread gDpMetaThread;

static void InitDpMetaQueue()
{
        if (gDpMetaQueue.size() != kDpMetaQueueSize)
                gDpMetaQueue.resize(kDpMetaQueueSize);
        gDpMetaHead.store(0, std::memory_order_release);
        gDpMetaTail.store(0, std::memory_order_release);
}

static bool EnqueueDpMeta(const EcInt& dist, int type)
{
        size_t head = gDpMetaHead.load(std::memory_order_relaxed);
        size_t next = (head + 1) & (kDpMetaQueueSize - 1);
        if (next == gDpMetaTail.load(std::memory_order_acquire))
                return false;
        gDpMetaQueue[head] = DpMeta{dist, type};
        gDpMetaHead.store(next, std::memory_order_release);
        return true;
}

static bool TryDequeueDpMeta(DpMeta& out)
{
        size_t tail = gDpMetaTail.load(std::memory_order_relaxed);
        if (tail == gDpMetaHead.load(std::memory_order_acquire))
                return false;
        out = gDpMetaQueue[tail];
        size_t next = (tail + 1) & (kDpMetaQueueSize - 1);
        gDpMetaTail.store(next, std::memory_order_release);
        return true;
}

static int GetBitLength(const EcInt& val)
{
        for (int i = 4; i >= 0; i--)
        {
                if (val.data[i])
                {
                        u32 index;
                        _BitScanReverse64((DWORD*)&index, val.data[i]);
                        return i * 64 + index + 1;
                }
        }
        return 0;
}

// Deserialize 22-byte DP distance into EcInt with sign extension if needed
static EcInt DeserializeDistance(const u8* dist)
{
        EcInt res;
        memcpy(res.data, dist, 22);
        // Sign-extend if negative marker is present
        if (dist[21] == 0xFF)
                memset(((u8*)res.data) + 22, 0xFF, 18);
        else
                memset(((u8*)res.data) + 22, 0, 18);
        return res;
}

// Absolute unsigned difference between two EcInt values
static EcInt AbsDistance(const EcInt& a, const EcInt& b)
{
        EcInt left = a;
        EcInt right = b;
        EcInt gap;
        if (left.IsLessThanU(right))
        {
                gap = right;
                gap.Sub(left);
        }
        else
        {
                gap = left;
                gap.Sub(right);
        }
        return gap;
}

// Normalize a candidate key so it always falls inside the configured search range
static EcInt NormalizeKeyToRange(const EcInt& cand)
{
        // If range width is unknown we cannot normalize
        if (gRangeWidth.IsZero())
                return cand;

        EcInt normalized = cand;

        // Work in coordinates relative to gStart to avoid overflow
        if (!gStart.IsZero())
        {
                bool borrowed = normalized.Sub(gStart);
                if (borrowed)
                        normalized.Add(gRangeWidth);
        }

        // Bring value into [0, gRangeWidth)
        while (!normalized.IsLessThanU(gRangeWidth))
                normalized.Sub(gRangeWidth);

        // Convert back to absolute coordinate
        if (!gStart.IsZero())
                normalized.Add(gStart);

        return normalized;
}

// Convert EcInt to billions using all limbs
static double EcIntToBillions(const EcInt& val)
{
        long double acc = 0.0L;
        for (int i = 4; i >= 0; --i)
        {
                        acc = acc * 18446744073709551616.0L + (long double)val.data[i];
        }
        acc /= 1000000000.0L;
        return (double)acc;
}

// Convert EcInt to full decimal string (unsigned interpretation)
static std::string EcIntToDecimal(const EcInt& val)
{
        u64 buffer[5];
        memcpy(buffer, val.data, sizeof(buffer));

        auto isZero = [&]() {
                for (int i = 0; i < 5; ++i)
                        if (buffer[i])
                                return false;
                return true;
        };

        if (isZero())
                return std::string("0");

        std::string result;
        while (!isZero())
        {
                u64 quotient[5] = {0, 0, 0, 0, 0};
                u64 rem = 0;
                for (int i = 4; i >= 0; --i)
                {
                        __uint128_t cur = ((__uint128_t)rem << 64) | buffer[i];
                        quotient[i] = (u64)(cur / 10);
                        rem = (u64)(cur % 10);
                }
                result.push_back((char)('0' + rem));
                memcpy(buffer, quotient, sizeof(buffer));
        }

        std::reverse(result.begin(), result.end());
        return result;
}

static bool IsLessThan(const EcInt& a, const EcInt& b)
{
        for (int i = 4; i >= 0; --i)
        {
                if (a.data[i] != b.data[i])
                        return a.data[i] < b.data[i];
        }
        return false;
}

static EcInt EstimateKeyFromPair(const DistanceEntry& a, const DistanceEntry& b)
{
        auto ApplyStartOffset = [](const EcInt& key) {
                EcInt withOffset = key;
                if (!gStart.IsZero())
                {
                        EcInt ofs = gStart;
                        withOffset.AddModP(ofs);
                }
                return withOffset;
        };

        EcInt bestKey;
        EcInt bestDist;
        bool hasBest = false;

        auto ConsiderCandidate = [&](const EcInt& candidate) {
                EcInt scalar = candidate;
                EcPoint P = ec.MultiplyG(scalar);
                if (P.IsEqual(gPntToSolve))
                {
                        bestKey = ApplyStartOffset(candidate);
                        return true;
                }

                EcInt withOffset = ApplyStartOffset(candidate);
                EcInt dist = AbsDistance(withOffset, gStart);
                if (!hasBest || dist.IsLessThanU(bestDist))
                {
                        bestKey = withOffset;
                        bestDist = dist;
                        hasBest = true;
                }
                return false;
        };

        auto EvaluateTameWild = [&](const EcInt& tameDist, const EcInt& wildDist, bool negateTame) {
                EcInt t = tameDist;
                EcInt w = wildDist;
                if (negateTame)
                        t.Neg();

                EcInt diff = t;
                diff.Sub(w);

                EcInt candidate = diff;
                candidate.Add(Int_HalfRange);
                if (ConsiderCandidate(candidate))
                        return true;

                EcInt altCandidate = diff;
                altCandidate.Neg();
                altCandidate.Add(Int_HalfRange);
                return ConsiderCandidate(altCandidate);
        };

        auto EvaluateWildPair = [&](const EcInt& firstWild, const EcInt& secondWild, bool negateFirst) {
                EcInt t = firstWild;
                EcInt w = secondWild;
                if (negateFirst)
                        t.Neg();

                EcInt diff = t;
                diff.Sub(w);
                if (diff.data[4] >> 63)
                        diff.Neg();

                diff.ShiftRight(1);

                EcInt candidate = diff;
                candidate.Add(Int_HalfRange);
                if (ConsiderCandidate(candidate))
                        return true;

                EcInt altCandidate = diff;
                altCandidate.Neg();
                altCandidate.Add(Int_HalfRange);
                return ConsiderCandidate(altCandidate);
        };

        bool aIsTame = a.type == TAME;
        bool bIsTame = b.type == TAME;

        if (aIsTame || bIsTame)
        {
                        EcInt tameDist = aIsTame ? a.dist : b.dist;
                        EcInt wildDist = aIsTame ? b.dist : a.dist;

                        if (EvaluateTameWild(tameDist, wildDist, false))
                                return bestKey;
                        EvaluateTameWild(tameDist, wildDist, true);
        }
        else
        {
                        if (EvaluateWildPair(a.dist, b.dist, false))
                                return bestKey;
                        EvaluateWildPair(a.dist, b.dist, true);
        }

        if (!hasBest)
        {
                bestKey.SetZero();
        }

        return bestKey;
}

static void UpdateGlobalGap(const DistanceEntry& distA, const DistanceEntry& distB)
{
        EcInt gap = AbsDistance(distA.dist, distB.dist);
        std::lock_guard<std::mutex> lock(gGapMutex);
        if (!gHasLowestGap || gap.IsLessThanU(gLowestGap))
        {
                gLowestGap = gap;
                gHasLowestGap = true;

                gBestDistanceA = distA;
                gBestDistanceB = distB;
                gHasGapPair = true;

                gEstimatedKey = EstimateKeyFromPair(distA, distB);
                gHasEstimatedKey = true;
        }
}

static void ConsiderGapWithSet(const DistanceEntry& entry, const std::multiset<DistanceEntry, DistanceEntryLess>& otherHerd)
{
        if (otherHerd.empty())
                return;

        auto it = otherHerd.lower_bound(entry);
        if (it != otherHerd.end())
                UpdateGlobalGap(entry, *it);

        if (it != otherHerd.begin())
        {
                --it;
                UpdateGlobalGap(entry, *it);
        }
}

static void ProcessDpMeta(const DpMeta& meta)
{
        if (gGenMode)
                return;

        if (meta.type == TAME)
                gTameCount.fetch_add(1, std::memory_order_relaxed);
        else if (meta.type == WILD1)
                gWild1Count.fetch_add(1, std::memory_order_relaxed);
        else if (meta.type == WILD2)
                gWild2Count.fetch_add(1, std::memory_order_relaxed);

        DistanceEntry entry{meta.dist, meta.type};
        if (meta.type == TAME)
        {
                gTameDistances.insert(entry);
                ConsiderGapWithSet(entry, gWild1Distances);
                ConsiderGapWithSet(entry, gWild2Distances);
        }
        else if (meta.type == WILD1)
        {
                gWild1Distances.insert(entry);
                ConsiderGapWithSet(entry, gTameDistances);
                ConsiderGapWithSet(entry, gWild2Distances);
        }
        else
        {
                gWild2Distances.insert(entry);
                ConsiderGapWithSet(entry, gTameDistances);
                ConsiderGapWithSet(entry, gWild1Distances);
        }
}

static void DpMetaConsumerProc()
{
        DpMeta meta;
        while (gDpMetaRunning.load(std::memory_order_acquire) ||
               gDpMetaTail.load(std::memory_order_acquire) != gDpMetaHead.load(std::memory_order_acquire))
        {
                if (TryDequeueDpMeta(meta))
                {
                        ProcessDpMeta(meta);
                        continue;
                }
                Sleep(1);
        }
}

static void StartDpMetaConsumer()
{
        InitDpMetaQueue();
        gDpMetaRunning.store(true, std::memory_order_release);
        gDpMetaThread = std::thread(DpMetaConsumerProc);
}

static void StopDpMetaConsumer()
{
        gDpMetaRunning.store(false, std::memory_order_release);
        if (gDpMetaThread.joinable())
                gDpMetaThread.join();
}

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

//	gcnt = 1; //dbg
	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
		
		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("\n\rDPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
}


void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

        for (int i = 0; i < cnt; i++)
        {
                DBRec nrec;
                u8* p = pPntList2 + i * GPU_DP_SIZE;
                memcpy(nrec.x, p, 12);
                memcpy(nrec.d, p + 16, 22);
                nrec.type = gGenMode ? TAME : p[40];

                EcInt fullDist = DeserializeDistance(nrec.d);

                if (!gGenMode)
                        EnqueueDpMeta(fullDist, nrec.type);

                DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
                if (gGenMode)
                        continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			if (pref->type == nrec.type)
			{
				if (pref->type == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
				//else
				//	ToLog("key found by same wild");
			}

			EcInt w, t;
			int TameType, WildType;
			if (pref->type != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = nrec.type;
				WildType = pref->type;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = nrec.type;
			}

			// Verify if this is a collision (matching X coordinate)
			bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
			if (!res)
			{
				bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
				if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
					;// ToLog("W1 and W2 collides in mirror");
				else
				{
					printf("\n\rCollision Error\r\n");
					gTotalErrors++;
				}
				continue;
			}

                        // Solution found! Use actual found key
                        {
                                std::lock_guard<std::mutex> lock(gGapMutex);
                                gEstimatedKey = gPrivKey;
                                if (!gStart.IsZero())
                                {
                                        EcInt ofs = gStart;
                                        gEstimatedKey.AddModP(ofs);
                                }
                                gHasEstimatedKey = true;
                        }

			gSolved = true;
			break;
		}
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val, u64 total_ops)
{
#ifdef DEBUG_MODE
        for (int i = 0; i <= MD_LEN; i++)
        {
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
	if (speed)
		exp_sec = (u64)((exp_ops / 1000000) / speed); //in sec
	u64 exp_days = exp_sec / (3600 * 24);
	int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

	u64 sec = (GetTickCount64() - tm_start) / 1000;
	u64 days = sec / (3600 * 24);
	int hours = (int)(sec - days * (3600 * 24)) / 3600;
	int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;

	u64 tameCount = gTameCount.load(std::memory_order_relaxed);
	u64 wild1Count = gWild1Count.load(std::memory_order_relaxed);
	u64 wild2Count = gWild2Count.load(std::memory_order_relaxed);
	u64 wildTotal = wild1Count + wild2Count;
	double twRatio = (wildTotal > 0) ? ((double)tameCount / (double)wildTotal) : 0.0;

	bool hasLowestGap = false;
	bool hasEstimatedKey = false;
	EcInt lowestGap;
	EcInt estimatedKey;
	{
		std::lock_guard<std::mutex> lock(gGapMutex);
		hasLowestGap = gHasLowestGap;
		hasEstimatedKey = gHasEstimatedKey;
		if (hasLowestGap)
			lowestGap = gLowestGap;
		if (hasEstimatedKey)
			estimatedKey = gEstimatedKey;
	}

        // Format lowest gap using full precision converted to billions
        char gapStr[100];
        if (hasLowestGap)
        {
                double gapDisplay = EcIntToBillions(lowestGap);
                snprintf(gapStr, sizeof(gapStr), "%.1f", gapDisplay);
        }
        else
        {
                sprintf(gapStr, "N/A");
        }

        // Format estimated key (full decimal, no trimming)
        char keyStr[200];
        if (hasEstimatedKey)
        {
                std::string decimal = EcIntToDecimal(estimatedKey);
                strncpy(keyStr, decimal.c_str(), sizeof(keyStr) - 1);
                keyStr[sizeof(keyStr) - 1] = 0;
        }
        else
        {
                sprintf(keyStr, "N/A");
        }

	double speed_mks = (double)speed;
	double count_log2 = 0.0;
	if (total_ops > 0)
		count_log2 = log2((double)total_ops);

	// Use carriage return for sticky progress bar (updates in place)
	printf("\r[%.2f MK/s][Count 2^%.2f][Dead %u][T/W:%.3f][L.Gap:%s][k_est:%s][%llud:%02dh:%02dm/%llud:%02dh:%02dm]  ",
		speed_mks,
		count_log2,
		gTotalErrors,
		twRatio,
		gapStr,
		keyStr,
		days, hours, min,
		exp_days, exp_hours, exp_min);
	fflush(stdout);
}

bool SolvePoint(EcPoint PntToSolve, EcInt& RangeWidth, int RangeBits, int DP, EcInt* pk_res)
{
        if ((RangeBits < 32) || (RangeBits > 180))
        {
                printf("Unsupported Range value (%d)!\r\n", RangeBits);
                return false;
        }
        if ((DP < 14) || (DP > 60))
        {
                printf("Unsupported DP value (%d)!\r\n", DP);
                return false;
        }

        int RangeWidthBits = GetBitLength(RangeWidth);
	printf("\r\nSolving point: Range %d bits (width bits %d), DP %d, start...\r\n", RangeBits, RangeWidthBits, DP);
        double ops = 1.15 * pow(2.0, RangeBits / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}

	u64 total_kangs = GpuKangs[0]->CalcKangCnt();
	for (int i = 1; i < GpuCnt; i++)
		total_kangs += GpuKangs[i]->CalcKangCnt();

	u64 dp_mask = ~((1ull << (64 - DP)) - 1);
	printf("Number of CPU thread: 0\r\n");
	printf("Range width: 2^%d\r\n", RangeBits);
	printf("Number of kangaroos: 2^%.2f\r\n", log2((double)total_kangs));
	printf("Suggested DP: %d\r\n", DP);
	printf("Expected operations: 2^%.2f\r\n", log2(ops));
	printf("Expected RAM: %.1fMB\r\n", ram * 1024.0);
	printf("DP size: %d [0x%016llX]\r\n", DP, (unsigned long long)dp_mask);
	double path_single_kang = ops / total_kangs;	
	double DPs_per_kang = path_single_kang / dp_val;
	printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

        if (!gGenMode && gTamesFileName[0])
        {
                printf("load tames...\r\n");
                if (db.LoadFromFile(gTamesFileName))
                {
                        printf("tames loaded\r\n");
                        if (db.Header[0] != gRangeBits)
                        {
                                printf("loaded tames have different range, they cannot be used, clear\r\n");
                                db.Clear();
                        }
                        else if (memcmp(db.Header + 4, gStart.data, 32) || memcmp(db.Header + 36, gEnd.data, 32))
                        {
                                printf("loaded tames have different start/end bounds, they cannot be used, clear\r\n");
                                db.Clear();
                        }
                }
                else
                        printf("tames loading failed\r\n");
        }

	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;

	// Initialize statistics
        gTameCount.store(0, std::memory_order_relaxed);
        gWild1Count.store(0, std::memory_order_relaxed);
        gWild2Count.store(0, std::memory_order_relaxed);
        {
                std::lock_guard<std::mutex> lock(gGapMutex);
                gLowestGap.SetZero();
                gHasLowestGap = false;
                gEstimatedKey.SetZero();
                gHasEstimatedKey = false;
                gHasGapPair = false;
        }
        gTameDistances.clear();
        gWild1Distances.clear();
        gWild2Distances.clear();
        gBestDistanceA.dist.SetZero();
        gBestDistanceA.type = 0;
        gBestDistanceB.dist.SetZero();
        gBestDistanceB.type = 0;
//prepare jumps
        EcInt minjump, t;
        minjump.Set(1);
        minjump.ShiftLeft(RangeBits / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
	}

        minjump.Set(1);
        minjump.ShiftLeft(RangeBits - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
	}

        minjump.Set(1);
        minjump.ShiftLeft(RangeBits - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
	}
	SetRndSeed(GetTickCount64());

        Int_HalfRange = RangeWidth;
        Int_HalfRange.ShiftRight(1);
        Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
        Pnt_NegHalfRange = Pnt_HalfRange;
        Pnt_NegHalfRange.y.NegModP();
        Int_TameOffset = Int_HalfRange;
        EcInt tt = RangeWidth;
        tt.ShiftRight(5); //half of tame range width
        Int_TameOffset.Sub(tt);
        gPntToSolve = PntToSolve;

//prepare GPUs
        for (int i = 0; i < GpuCnt; i++)
                if (!GpuKangs[i]->Prepare(PntToSolve, RangeBits, DP, RangeWidth, EcJumps1, EcJumps2, EcJumps3))
                {
			GpuKangs[i]->Failed = true;
			printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
		}

	u64 tm0 = GetTickCount64();
	printf("GPUs started...\r\n");

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT];
#else
	pthread_t thr_handles[MAX_GPU_CNT];
#endif

	u32 ThreadID;
	gSolved = false;
	StartDpMetaConsumer();
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(10);
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val, PntTotalOps);
			tm_stats = GetTickCount64();
		}

		if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
		{
			gIsOpsLimit = true;
			printf("\n\rOperations limit reached\r\n");
			printf("Aborted !\r\n");
			break;
		}
	}

	printf("\n\rStopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(10);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}
	StopDpMetaConsumer();

	if (gIsOpsLimit)
	{
                if (gGenMode)
                {
                        printf("saving tames...\r\n");
                        memset(db.Header, 0, sizeof(db.Header));
                        db.Header[0] = gRangeBits;
                        memcpy(db.Header + 4, gStart.data, 32);
                        memcpy(db.Header + 36, gEnd.data, 32);
                        if (db.SaveToFile(gTamesFileName))
                                printf("tames saved\r\n");
                        else
                                printf("tames saving failed\r\n");
                }
		db.Clear();
		return false;
	}

        double K = (double)PntTotalOps / pow(2.0, RangeBits / 2.0);
	printf("\n\rPoint solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
	db.Clear();
	*pk_res = gPrivKey;
	return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if ((strcmp(argument, "-h") == 0) || (strcmp(argument, "--help") == 0))
		{
			printf("Usage: RCKangaroo [options]\r\n");
			printf("Options:\r\n");
			printf("  -gpu <mask>                 GPU indices mask, e.g. 012\r\n");
			printf("  -d <bits>                   DP bits (14..60)\r\n");
			printf("  --start-hex <hex>           Range start (hex)\r\n");
			printf("  --end-hex <hex>             Range end (hex)\r\n");
			printf("  --start-dec <dec>           Range start (decimal)\r\n");
			printf("  --end-dec <dec>             Range end (decimal)\r\n");
			printf("  -pubkey <hex>               Public key (hex)\r\n");
			printf("  --pubkey <hex>              Public key (hex)\r\n");
			printf("  -tames <file>               Tames filename\r\n");
			printf("  -m <value>                  Max ops limit (for tames generation)\r\n");
			printf("  -h, --help                  Show this help\r\n");
			return false;
		}
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-d") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -d option\r\n");
				return false;
			}
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 14) || (val > 60))
			{
				printf("error: invalid value for -d option\r\n");
				return false;
			}
			gDP = val;
		}
		else
                if (strcmp(argument, "--start-hex") == 0)
                {
			if (ci >= argc)
			{
				printf("error: missed value after --start-hex option\r\n");
				return false;
			}
			if (gStartSet)
			{
				printf("error: start range already specified\r\n");
				return false;
			}
                        if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for --start-hex option\r\n");
				return false;
			}
                        ci++;
                        gStartSet = true;
                }
                else
                if (strcmp(argument, "--end-hex") == 0)
                {
			if (ci >= argc)
			{
				printf("error: missed value after --end-hex option\r\n");
				return false;
			}
			if (gEndSet)
			{
				printf("error: end range already specified\r\n");
				return false;
			}
                        if (!gEnd.SetHexStr(argv[ci]))
                        {
                                printf("error: invalid value for --end-hex option\r\n");
                                return false;
                        }
                        ci++;
                        gEndSet = true;
                }
		else
                if (strcmp(argument, "--start-dec") == 0)
                {
			if (ci >= argc)
			{
				printf("error: missed value after --start-dec option\r\n");
				return false;
			}
			if (gStartSet)
			{
				printf("error: start range already specified\r\n");
				return false;
			}
                        if (!gStart.SetDecStr(argv[ci]))
			{
				printf("error: invalid value for --start-dec option\r\n");
				return false;
			}
                        ci++;
                        gStartSet = true;
                }
                else
                if (strcmp(argument, "--end-dec") == 0)
                {
			if (ci >= argc)
			{
				printf("error: missed value after --end-dec option\r\n");
				return false;
			}
			if (gEndSet)
			{
				printf("error: end range already specified\r\n");
				return false;
			}
                        if (!gEnd.SetDecStr(argv[ci]))
                        {
                                printf("error: invalid value for --end-dec option\r\n");
                                return false;
                        }
                        ci++;
                        gEndSet = true;
                }
                else
                if (strcmp(argument, "-pubkey") == 0 || strcmp(argument, "--pubkey") == 0)
                {
			if (ci >= argc)
			{
				printf("error: missed value after --pubkey option\r\n");
				return false;
			}
                        if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for --pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -tames option\r\n");
				return false;
			}
			strcpy(gTamesFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-m") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -m option\r\n");
				return false;
			}
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -m option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
        if (!gPubKey.x.IsZero())
                if (!gStartSet || !gEndSet || !gDP)
                {
                        printf("error: you must also specify -d and a start/end range options\r\n");
                        return false;
                }
        if (gStartSet && gEndSet)
        {
                gRangeWidth = gEnd;
                bool carry = gRangeWidth.Sub(gStart);
                if (carry || gRangeWidth.IsZero())
                {
                        printf("error: end must be greater than start\r\n");
                        return false;
                }
                gRangeBits = GetBitLength(gRangeWidth);
                if ((gRangeBits < 32) || (gRangeBits > 170))
                {
                        printf("error: start/end range width must be between 32 and 170 bits\r\n");
                        return false;
                }
        }
        if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
        {
                if (gMax == 0.0)
                {
                        printf("error: you must also specify -m option to generate tames\r\n");
                        return false;
                }
                gGenMode = true;
                if (!gStartSet || !gEndSet)
                {
                        printf("error: start/end options are required when generating tames\r\n");
                        return false;
                }
        }
        return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	printf("********************************************************************************\r\n");
	printf("*                    RCKangaroo v3.1  (c) 2024 RetiredCoder                    *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
        printf("DEBUG MODE\r\n\r\n");
#endif

        InitEc();
        gDP = 0;
        gRangeBits = 0;
        gStart.SetZero();
        gStartSet = false;
        gEnd.SetZero();
        gEndSet = false;
        gRangeWidth.SetZero();
        gTamesFileName[0] = 0;
        gMax = 0.0;
        gGenMode = false;
        gIsOpsLimit = false;
	memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
	if (!ParseCommandLine(argc, argv))
		return 0;

	InitGpus();

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
			PntOfs = ec.MultiplyG(gStart);
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}

                char sx[100], sy[100];
                gPubKey.x.GetHexStr(sx);
                gPubKey.y.GetHexStr(sy);
                printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
                gStart.GetHexStr(sx);
                printf("Offset: %s\r\n", sx);
                gEnd.GetHexStr(sx);
                printf("End: %s\r\n", sx);

                if (!SolvePoint(PntToSolve, gRangeWidth, gRangeBits, gDP, &pk_found))
                {
                        if (!gIsOpsLimit)
                                printf("FATAL ERROR: SolvePoint failed\r\n");
			goto label_end;
		}
		pk_found.AddModP(gStart);
		EcPoint tmp = ec.MultiplyG(pk_found);
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n", s);
		printf("Priv: 0x%s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");
		//solve points, show K
                while (1)
                {
                        EcInt pk, pk_found;
                        EcPoint PntToSolve;

                        if (!gRangeBits)
                                gRangeBits = 78;
                        if (gRangeWidth.IsZero())
                        {
                                gRangeWidth.Set(1);
                                gRangeWidth.ShiftLeft(gRangeBits);
                                gEnd = gRangeWidth;
                        }
                        if (!gDP)
                                gDP = 16;

                        //generate random pk
                        pk.RndBits(gRangeBits);
                        PntToSolve = ec.MultiplyG(pk);

                        if (!SolvePoint(PntToSolve, gRangeWidth, gRangeBits, gDP, &pk_found))
                        {
                                if (!gIsOpsLimit)
                                        printf("FATAL ERROR: SolvePoint failed\r\n");
				break;
			}
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
                        }
                        TotalOps += PntTotalOps;
                        TotalSolved++;
                        u64 ops_per_pnt = TotalOps / TotalSolved;
                        double K = (double)ops_per_pnt / pow(2.0, gRangeBits / 2.0);
                        printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
	free(pPntList2);
	free(pPntList);
}


