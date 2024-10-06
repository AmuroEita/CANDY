/**
 * @Author: Junyao Dong
 * @Date: 10/1/2024, 7:19:37 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/1/2024, 7:19:37 PM
 * Description: 
 */

#ifndef CANDY_FUNC_STRATEGY_H
#define CANDY_FUNC_STRATEGY_H

#include <functional>
#include <mutex>
#include <condition_variable>
#include <Utils/IntelliTensorOP.hpp>
#include <libco/co_routine.h>

class ThreadManager {
protected:
	int wThreadsCnt = 0; 
	int rThreadsCnt = 0;
	bool enableCoroutine = false;

	// wether or not the read threads share the same tensor
	bool readShared = true;
	
	// wether or not the write threads share the same tensor
	bool writeShared = true;

public:
	ThreadManager(int threads = 1, int readThreads = 0, bool en = false) : 
		threadCnt(threads), readThreadCnt(readThreads), enableCoroutine(en) {}
	
	~ThreadManager();
	
	void setConfig();
	
	virtual void initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) = 0;
	
	virtual void initSearchTensor(const std::function<std::vector(torch::Tensor&)>& func, torch::Tensor &t, int64_ k) = 0;
	
	// TODO
	virtual void initDeleteTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t);
	
	virtual void exec() = 0;
};

class MultiThreadStrategy : public ThreadManager {
	std::mutex mtx;
	std::condition_variable cv;
	bool ready = false;
	
public:
	MultiThreadManager() : ThreadManager() {}

	void initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) override;
	
	void initSearchTensor(const std::function<std::vector(torch::Tensor&)>& func, torch::Tensor &t, int64_ k) override;
	
	// void initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) override;
	
	vold exec();
};

class CoroutineManager : public ThreadManager {
public:
	CoroutineManager() : ThreadManager() {}

	void initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) override;
	
	void initSearchTensor(const std::function<std::vector(torch::Tensor&)>& func, torch::Tensor &t, int64_ k) override;
	
	// void initInsertTensor(const std::function<bool(torch::Tensor&)>& func, torch::Tensor &t) override;
	
	void exec();
};

#endif