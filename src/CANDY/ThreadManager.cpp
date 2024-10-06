/**
 * @Author: Junyao Dong
 * @Date: 10/1/2024, 7:37:15 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/1/2024, 7:37:15 PM
 * Description: 
 */

#include <thread>
#include <atomic>
#include <CANDY/ThreadManager.h>

std::atomic<bool> readyFlag = false;

struct CoInsertTensorArgs {
	const std::function<bool(torch::Tensor&)>& func;
	torch::Tensor tensor;
};

struct CoSearchTensorArgs {
	const std::function<bool(torch::Tensor&)>& func;
	torch::Tensor tensor;
	int64_t k;
};

void MultiThreadStrategy::initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) {
	std::vector<std::thread> threads(wThreadCnt);
	for (int i = 0; i < wThreadCnt; i++) {
		threads.emplace_back(
			std::thread([this, &func, &t]() {
				std::unique_lock<std::mutex> lock(this->mtx);
				this->cv.wait(lock, [this]{ return this->ready; });  
				func(t);  
		})); 
	} 
	return;
}

void MultiThreadStrategy::initSearchTensor(
	const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t, int64_t k) {
	std::vector<std::thread> threads(rThreadCnt);
	for (int i = 0; i < rThreadCnt; i++) {
		threads.emplace_back(
			std::thread([this, &func, &t, k]() {
				std::unique_lock<std::mutex> lock(this->mtx);
				this->cv.wait(lock, [this]{ return this->ready; });  
				func(t);  
		})); 
	} 
}

void MultiThreadStrategy::exec() { 
	{
		std::lock_guard<std::mutex> lock(mtx);
		ready = true;
	}
	cv.notify_all(); 
}

void* CoInsertTensor(void* args) {
	while (!readyFlag.load()) { 
		co_yield_ct();
	}
	
	auto* coArgs = static_cast<CoInsertTensorArgs*>(args);
	coArgs->func(coArgs->tensor);  
	return nullptr;
}

void CoroutineManager::initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) {
	CoInsertTensorArgs coArg(func, t);
	std::vector<stCoRoutine_t*> co(wThreadCnt);
	 
	for (int i = 0; i < threadCnt; i++) { 
		co_create(&co[i], nullptr, CoInsertTensor, &coArg);
		co_resume(co[i]);
	}
}

void* CoSearchTensor(void* args) {
	while (!readyFlag.load()) { 
		co_yield_ct();
	}
	
	auto* coArgs = static_cast<CoInsertTensorArgs*>(args);
	coArgs->func(coArgs->tensor, args->k);  
	return nullptr;
}

void CoroutineManager::initSearchTensor(
	const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t, int64_t k) {
	CoSearchTensorArgs coArg(func, t, k);
	std::vector<stCoRoutine_t*> co(rThreadCnt);
	 
	for (int i = 0; i < threadCnt; i++) { 
		co_create(&co[i], nullptr, CoInsertTensor, &coArg);
		co_resume(co[i]);
	}
}

void CoroutineManager::exec() { 
	readyFlag.store(true);
	co_eventloop(co_get_epoll_ct(), nullptr, nullptr);	
}