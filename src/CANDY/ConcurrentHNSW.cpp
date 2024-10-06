/**
 * @Author: Junyao Dong
 * @Date: 10/3/2024, 4:45:37 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/3/2024, 4:45:37 PM
 * Description: 
 */

#include <CANDY/ThreadManager.h>
#include <CANDY/ConcurrentHNSWIndex.h>

bool CANDY::ConcurrentHNSWIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
	AbstractIndex::setConfig(cfg);
	assert(cfg);
	
	is_NSW = cfg->tryI64("is_NSW", 0, true);
	vecDim = cfg->tryI64("vecDim", 768, true);
	M_ = cfg->tryI64("maxConnection", 32, true);
	std::string metricType = cfg->tryString("metricType", "L2", true);
	faissMetric = faiss::METRIC_L2;
	if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
			faissMetric = faiss::METRIC_INNER_PRODUCT;
	}
	hnsw = HNSW(vecDim, M_);

	opt_mode_ = cfg->tryI64("opt_mode", 0, true);
	hnsw.set_mode(opt_mode_, faissMetric);

	if(opt_mode_ == OPT_DCO){
		adSampling_step = cfg->tryI64("samplingStep", 64, true);
		adSampling_epsilon0 = cfg->tryDouble("ads_epsilon", 1.0,true);
		printf("adSampling_step = %ld\n", adSampling_step);
	}
	
	thMgr = new ThreadManager();
	thMgr->setConfig(cfg);

	storage = new CANDY::FlatIndex();
	storage->setConfig(cfg);
	return true;
}