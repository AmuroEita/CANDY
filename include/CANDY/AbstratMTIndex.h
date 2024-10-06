/**
 * @Author: Junyao Dong
 * @Date: 10/3/2024, 2:42:24 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/3/2024, 2:42:24 PM
 * Description: 
 */

#ifndef CANDY_INCLUDE_CANDY_ABSTRACTMTINDEX_H_
#define CANDY_INCLUDE_CANDY_ABSTRACTMTINDEX_H_

namespace CANDY {
	
class AbstractMTIndex {
	
	virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

	/**
	 * @brief revise a tensor
	 * @param t the tensor to be revised
	 * @param w the revised value
	 * @return bool whether the revising is successful
	 */
	virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w);
	/**
	 * @brief search the k-NN of a query tensor, return their index
	 * @param t the tensor, allow multiple rows
	 * @param k the returned neighbors
	 * @return std::vector<faiss::idx_t> the index, follow faiss's order
	 */
	virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);	
};
	
}