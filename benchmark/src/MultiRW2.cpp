/**
 * @Author: Junyao Dong
 * @Date: 9/30/2024, 9:52:55 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 9/30/2024, 11:50:08 PM
 * Description: Benchmarking for multiple read(serach) and write(insert)
 */

#include <iostream>
#include <torch/torch.h>
#include <CANDY.h>
#include <Utils/UtilityFunctions.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

using namespace INTELLI;
static inline CANDY::AbstractIndexPtr indexPtr = nullptr;
static inline std::vector<INTELLI::IntelliTimeStampPtr> timeStamps;
static inline timer_t timerid;

bool fileExists(const std::string &filename) {
	std::ifstream file(filename);
	return file.good(); // Returns true if the file is open and in a good state
}
static inline int64_t s_timeOutSeconds = -1;

static inline void earlyTerminateTimerCallBack() {
	INTELLI_ERROR(
		"Force to terminate due to timeout in " + std::to_string(s_timeOutSeconds) + "seconds");
	auto briefOutCfg = newConfigMap();
	double latency95 = 0;
	briefOutCfg->edit("throughput", (int64_t) 0);
	briefOutCfg->edit("recall", (int64_t) 0);
	briefOutCfg->edit("throughputByElements", (int64_t) 0);
	briefOutCfg->edit("95%latency(Insert)", latency95);
	briefOutCfg->edit("pendingWrite", latency95);
	briefOutCfg->edit("latencyOfQuery", (int64_t) 0);
	briefOutCfg->edit("normalExit", (int64_t) 0);
	briefOutCfg->toFile("onlineInsert_result.csv");
	std::cout << "brief results\n" << briefOutCfg->toString() << std::endl;
	exit(-1);
}

static inline void timerCallback(union sigval v) {
	earlyTerminateTimerCallBack();
}

static inline void setEarlyTerminateTimer(int64_t seconds) {
	struct sigevent sev;
	memset(&sev, 0, sizeof(struct sigevent));
	sev.sigev_notify = SIGEV_THREAD;
	sev.sigev_notify_function = timerCallback;
	struct itimerspec its;
	its.it_value.tv_sec = seconds;
	its.it_value.tv_nsec = 0;
	its.it_interval.tv_sec = seconds;
	its.it_interval.tv_nsec = 0;
	timer_create(CLOCK_REALTIME, &sev, &timerid);
	timer_settime(timerid, 0, &its, nullptr);
}

int main(int argc, char **argv) {

	// 1. Load the configs
	INTELLI::ConfigMapPtr inMap = newConfigMap();
	if (inMap->fromCArg(argc, argv) == false) {
		if (argc >= 2) {
			std::string fileName = "";
			fileName += argv[1];
			if (inMap->fromFile(fileName)) 
				INTELLI_INFO("Load config from file " + fileName);
		}
  	}
  
	// 2. Create the data and query, and prepare initialTensor
	CANDY::DataLoaderTable dataLoaderTable;
	std::string dataLoaderTag = inMap->tryString("dataLoaderTag", "random", true);
	int64_t cutOffTimeSeconds = inMap->tryI64("cutOffTimeSeconds", -1, true);
	int64_t waitPendingWrite = inMap->tryI64("waitPendingWrite", 0, true);
	s_numberOfRWSeq = inMap->tryI64("numberOfRWSeq", 1, true);
	throughputVec = std::vector<double>(s_numberOfRWSeq, 0);
	insert95Vec = std::vector<int64_t>(s_numberOfRWSeq, 0);
	pendingWriteVec = std::vector<int64_t>(s_numberOfRWSeq, 0);
	latQueryVec = std::vector<int64_t>(s_numberOfRWSeq, 0);
	recallVec = std::vector<double>(s_numberOfRWSeq, 0);
	auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
	INTELLI_INFO("Data loader : " + dataLoaderTag);
	if (dataLoader == nullptr) {
		return -1;
	}
	dataLoader->setConfig(inMap);
	int64_t initialRows = inMap->tryI64("initialRows", 0, true);
	auto dataTensorAll = dataLoader->getData().nan_to_num(0);
	
	auto dataTensorInitial = dataTensorAll.slice(0, 0, initialRows);
	auto dataTensorStreamAll = dataTensorAll.slice(0, initialRows, dataTensorAll.size(0));
	auto queryTensorAll = dataLoader->getQuery().nan_to_num(0);
	
	int64_t currSeq = 0;
	int64_t dbSeqRows = dataTensorStreamAll.size(0) / s_numberOfRWSeq;
	int64_t querySeqRows = queryTensorAll.size(0) / s_numberOfRWSeq;
	int64_t ANNK = inMap->tryI64("AKNN", 5, true);
	int64_t pendingWriteTime = 0;	
  
	// 3. create index
	CANDY::IndexTable indexTable;
	std::string indexTag = inMap->tryString("indexTag", "hnsw", true);
	indexPtr = indexTable.getIndex(indexTag);
	if (indexPtr == nullptr) {
		return -1;
	}
	
	// 4. streaming feed
	uint64_t startRow = 0;
	uint64_t endRow = startRow + batchSize;
	uint64_t tNow = 0;
	uint64_t tETA = timeStamps[endRow - 1]->arrivalTime;
	uint64_t tp = 0;
	uint64_t tDone = 0;
	uint64_t allRows = dataTensorStream.size(0);
	
	if (initialRows > 0) {
		indexPtr->loadInitialTensor(dataTensorInitial);
  	}
	
	auto start = std::chrono::high_resolution_clock::now();
	int64_t frozenLevel = inMap->tryI64("frozenLevel", 1, true);
	indexPtr->setFrozenLevel(frozenLevel);
	
	while (startRow < allRows) {
		tNow = chronoElapsedTime(start);
		while (tNow < tETA) 
	  		tNow = chronoElapsedTime(start);
				
		auto subTensor = dataTensorStream.slice(0, startRow, endRow);
		indexPtr->insertTensor(subTensor);
		tp = chronoElapsedTime(start);
		
		for (size_t i = startRow; i < endRow; i++) {
			timeStamps[i]->processedTime = tp;
		}
		
		startRow += batchSize;
		endRow += batchSize;
		if (endRow >= allRows)
			endRow = allRows;
			
		tETA = timeStamps[endRow - 1]->arrivalTime;
	}
	tDone = chronoElapsedTime(start);
	
	if (waitPendingWrite) {
		INTELLI_WARNING("There is pending write, wait first");
		auto startWP = std::chrono::high_resolution_clock::now();
		indexPtr->waitPendingOperations();
		pendingWriteTime = chronoElapsedTime(startWP);
		INTELLI_INFO("Wait " + std::to_string(pendingWriteTime / 1000) + " ms for pending writings");
	}

	INTELLI_INFO("Insert is done, let us validate the results");
    auto startQuery = std::chrono::high_resolution_clock::now();
    auto indexResults = indexPtr->searchTensor(queryTensor, ANNK);
    tNow = chronoElapsedTime(startQuery);
    INTELLI_INFO("Query done in " + to_string(tNow / 1000) + "ms");
    uint64_t queryLatency = tNow;

    std::string groundTruthPrefix = inMap->tryString("groundTruthPrefix", "onlineInsert_GroundTruth", true);

    std::string probeName = groundTruthPrefix + "/" + std::to_string(indexResults.size() - 1) + ".rbt";
    double recall = 0.0;
		
	{
		INTELLI_INFO("Ground truth does not exist, so I'll create it");

		gdIndex->insertTensor(dataTensorStream);

		auto gdResults = gdIndex->searchTensor(queryTensor, ANNK);
		INTELLI_INFO("Ground truth is done");
		recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
		UtilityFunctions::tensorListToFile(gdResults, groundTruthPrefix);
    }

		double throughput = aRows * 1e6 / tDone;
		throughputVec[currSeq] = throughput;
		pendingWriteVec[currSeq] = pendingWriteTime;
		insert95Vec[currSeq] = UtilityFunctions::getLatencyPercentage(0.95, timeStamps);
		recallVec[currSeq] = recall;
		latQueryVec[currSeq] = queryLatency;
		UtilityFunctions::saveTimeStampToFile("multiRW_timestamps" + to_string(currSeq) + ".csv", timeStamps);
	}

	indexPtr->endHPC();
	indexPtr->isHPCStarted = false;
	auto briefOutCfg = newConfigMap();
	generateNormalResultCsv(briefOutCfg, "multiRW_result.csv");
	return 0;
}