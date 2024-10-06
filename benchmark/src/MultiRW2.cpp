/**
 * @Author: Junyao Dong
 * @Date: 9/30/2024, 9:52:55 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/1/2024, 4:21:52 PM
 * Description: Benchmarking for multiple query and insert, only support HNSW in this version.
 * 
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
	auto briefOutconf = newConfigMap();
	double latency95 = 0;
	briefOutconf->edit("throughput", (int64_t) 0);
	briefOutconf->edit("recall", (int64_t) 0);
	briefOutconf->edit("throughputByElements", (int64_t) 0);
	briefOutconf->edit("95%latency(Insert)", latency95);
	briefOutconf->edit("pendingWrite", latency95);
	briefOutconf->edit("latencyOfQuery", (int64_t) 0);
	briefOutconf->edit("normalExit", (int64_t) 0);
	briefOutconf->toFile("onlineInsert_result.csv");
	std::cout << "brief results\n" << briefOutconf->toString() << std::endl;
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
	INTELLI::ConfigMapPtr conf = newConfigMap();
	if (conf->fromCArg(argc, argv) == false) {
		if (argc >= 2) {
			std::string fileName = "";
			fileName += argv[1];
			if (conf->fromFile(fileName)) {
				INTELLI_INFO("Load config from file " + fileName);
			}
		}
  	}
  
	CANDY::DataLoaderTable dataLoaderTable;
	std::string dataLoaderTag = conf->tryString("dataLoaderTag", "random", true);
	int64_t cutOffInSec = conf->tryI64("cutOffTimeInSec", -1, true);
	int64_t waitPendingWrite = conf->tryI64("waitPendingWrite", 0, true);	
	int64_t initialRows = conf->tryI64("initialRows", 0, true);

	// 2. Create the data and query, and prepare initialTensor	
	auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
	// INTELLI_INFO("Data loader : %s.", dataLoaderTag);
	if (dataLoader == nullptr) {
		return -1;
	}
	
	dataLoader->setConfig(conf);
	auto dataTensorAll = dataLoader->getData().nan_to_num(0);
	auto dataTensorInitial = dataTensorAll.slice(0, 0, initialRows);
	auto dataTensorStream = dataTensorAll.slice(0, initialRows, dataTensorAll.size(0));
	auto queryTensor = dataLoader->getQuery().nan_to_num(0);
	int64_t batchSize = conf->tryI64("batchSize", dataTensorStream.size(0), true);
	
	// INTELLI_INFO(
	// 	"Initial tensor: Demension = %s, data = %s.",
	// 	std::to_string(dataTensorInitial.size(1)), 
	// 	std::to_string(dataTensorInitial.size(0)));
  	// INTELLI_INFO(
	//   	"Streaming tensor: Demension = %s, data = %s, query = %s", 
	// 	std::to_string(dataTensorStream.size(1)),
	// 	std::to_string(dataTensorStream.size(0)),
	// 	std::to_string(queryTensor.size(0)));
	
	// 3. create the timestamps
	INTELLI::IntelliTimeStampGenerator timeStampGen;
	conf->edit("streamingTupleCnt", (int64_t) dataTensorStream.size(0));
	timeStampGen.setConfig(conf);
	timeStamps = timeStampGen.getTimeStamps();
	// INTELLI_INFO("TimeStampSize = " + std::to_string(timeStamps.size()));
	
	// 4. create index
	// Concurrency control benchmarking only supports HNSW in this version.
	CANDY::IndexTable indexTable;
	std::string indexTag = conf->tryString("indexTag", "hnsw", true);
	indexPtr = indexTable.getIndex(indexTag);
	if (indexPtr == nullptr) {
		return -1;
	}
	indexPtr->setConfig(conf);
	
	// 5. streaming feed
	uint64_t startRow = 0;
	uint64_t endRow = startRow + batchSize;
	uint64_t tNow = 0;
	uint64_t tETA = timeStamps[endRow - 1]->arrivalTime;
	uint64_t tp = 0;
	uint64_t tDone = 0;
	uint64_t allRows = dataTensorStream.size(0);
	
	INTELLI_INFO("Load initial tensor.");
	if (initialRows > 0) {
		indexPtr->loadInitialTensor(dataTensorInitial);
  	}
	
	auto start = std::chrono::high_resolution_clock::now();
	
	while (startRow < allRows) {
		tNow = chronoElapsedTime(start);
		while (tNow < tETA) 
	  		tNow = chronoElapsedTime(start);
				
		auto subTensor = dataTensorStream.slice(0, startRow, endRow);
		
		// Include read and write tasks 
		// indexPtr->insertTensor(subTensor);
		// indexPtr->searchTensor(subTensor);
		
		// indexPtr->initInsertTensor(subTensor);
		// indexPtr->initSearchTensor(subTensor);
		
		indexPtr->exec();
		
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
	
	// if (waitPendingWrite) {
	// 	INTELLI_WARNING("There is pending write, wait first.");
	// 	auto startWP = std::chrono::high_resolution_clock::now();
	// 	indexPtr->waitPendingOperations();
	// 	pendingWriteTime = chronoElapsedTime(startWP);
	// 	INTELLI_INFO("Wait " + std::to_string(pendingWriteTime / 1000) + " ms for pending writings");
	// }

	// 5. preocess results
	INTELLI_INFO("Insert is done, let us validate the results");
	// auto startQuery = std::chrono::high_resolution_clock::now();
	// auto indexResults = indexPtr->searchTensor(queryTensor, ANNK);
	// tNow = chronoElapsedTime(startQuery);
	// INTELLI_INFO("Query done in " + to_string(tNow / 1000) + "ms");
	// uint64_t queryLatency = tNow;

	// std::string groundTruthPrefix = inMap->tryString("groundTruthPrefix", "onlineInsert_GroundTruth", true);

	// std::string probeName = groundTruthPrefix + "/" + std::to_string(indexResults.size() - 1) + ".rbt";
	// double recall = 0.0;
		
	// {
	// 	INTELLI_INFO("Ground truth does not exist, so I'll create it");

	// 	gdIndex->insertTensor(dataTensorStream);

	// 	auto gdResults = gdIndex->searchTensor(queryTensor, ANNK);
	// 	INTELLI_INFO("Ground truth is done");
	// 	recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
	// 	UtilityFunctions::tensorListToFile(gdResults, groundTruthPrefix);
	// }

	// double throughput = aRows * 1e6 / tDone;
	// throughputVec[currSeq] = throughput;
	// pendingWriteVec[currSeq] = pendingWriteTime;
	// insert95Vec[currSeq] = UtilityFunctions::getLatencyPercentage(0.95, timeStamps);
	// recallVec[currSeq] = recall;
	// latQueryVec[currSeq] = queryLatency;
	// UtilityFunctions::saveTimeStampToFile("multiRW_timestamps" + to_string(currSeq) + ".csv", timeStamps);

	// indexPtr->endHPC();
	// indexPtr->isHPCStarted = false;
	// auto briefOutconf = newConfigMap();
	// generateNormalResultCsv(briefOutconf, "multiRW2_result.csv");
	return 0;
}