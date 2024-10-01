/**
 * @Author: Junyao Dong
 * @Date: 10/1/2024, 7:19:37 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/1/2024, 7:19:37 PM
 * Description: 
 */

#include <functional>
#include <Utils/IntelliTensorOP.hpp>

class FuncStrategy {
protected:
	int threadCnt; 
	int readThreadCnt;
	bool enableCoroutine;

public:
	Functrategy(int threads = 1, int readThreads = 0, bool en = false) : 
		threadCnt(threads), readThreadCnt(readThreads), enableCoroutine(en) {}
	
	virtual ~FuncStrategy();
	
	virtual void exec(const std::function<void()>& insertFunc, torch::Tensor &t) = 0;
};

class SingleThreadFunc : public FuncStrategy {
public:
	SingleThreadFunc() : FuncStrategy() {
		
	}
	
	void exec(const std::function<void()>& func, torch::Tensor &t) override;
}

// TODO
// class MultiThreadInsert : public InsertStrategy {
// public:
// 	void insertTensor() override;
// };

// class CoroutineInsert : public InsertStrategy {
// public:
// 	void insertTensor() override;
// };
