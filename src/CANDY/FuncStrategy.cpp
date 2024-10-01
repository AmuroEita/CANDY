/**
 * @Author: Junyao Dong
 * @Date: 10/1/2024, 7:37:15 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/1/2024, 7:37:15 PM
 * Description: 
 */

#include <CANDY/FuncStrategy.h>

void SingleThreadFunc::exec(const std::function<void()>& func, torch::Tensor &t) {
    func(t); 
}

// void SingleThreadFunc::exec() {
	
// }

// void SingleThreadFunc::exec() {
	
// }