/**
 * @Author: Junyao Dong
 * @Date: 10/3/2024, 4:49:57 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/3/2024, 4:49:57 PM
 * Description: 
 */

#ifndef CANDY_HNSWNAIVEINDEX_H
#define CANDY_HNSWNAIVEINDEX_H
#include <CANDY/AbstractIndex.h>
#include <CANDY/FlatIndex.h>
#include <CANDY/ConcurrentHNSW/ConcurrentHNSW.h>

namespace CANDY {
class HNSWConcurrentHNSWIndex : public AbstractIndex {
public:
	HNSW hnsw;
	bool isNSW;

	bool is_local_lvq = true;
	FlatIndex *storage = nullptr;
	INTELLI::ConfigMapPtr myCfg = nullptr;

	typedef int64_t opt_mode_t;
	opt_mode_t opt_mode_ = OPT_VANILLA;
	faiss::MetricType faissMetric = faiss::METRIC_L2;

	int64_t vecDim;
	
	/// Number of neighbors in HNSW structure
	int64_t M_ = 32;
	
	/// Number of all vectors
	int64_t ntotal = 0;

	int64_t adSampling_step = 32;
	float adSampling_epsilon0 = 1.0;
	
	// enum class CcControlVersion {BASELINE, FAISS, MILVUS, HNSWLIB};
	// CcControlVersion ccVer;
	
	ThreadManager thMgr;

	ConcurrentHNSWIndex(){};

	virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

	virtual bool insertTensor(torch::Tensor &t);

	virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

	virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
};

#define newConcurrentHNSWIndex std::make_shared<CANDY::ConcurrentHNSWIndex>
#define newConcurrentNSWIndex std::make_shared<CANDY::ConcurrentHNSWIndex>

} 

#endif
