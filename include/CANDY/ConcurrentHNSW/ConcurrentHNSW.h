/**
 * @Author: Junyao Dong
 * @Date: 10/6/2024, 2:12:26 PM
 * @LastEditors: Junyao Dong
 * @LastEditTime: 10/6/2024, 2:12:26 PM
 * Description: 
 */

#ifndef CANDY_CONCURRENT_HNSW_H
#define CANDY_CONCURRENT_HNSW_H
#include <CANDY/HNSWNaive/DistanceQueryer.h>
#include <Utils/IntelliTensorOP.hpp>
#include <faiss/MetricType.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <queue>
#include <string>
#include <vector>

typedef std::vector<INTELLI::TensorPtr> TensorVec;
#define NULL_NEIGHBOR = nullptr;
namespace CANDY {
class HNSWVertex {
public:
	INTELLI::TensorPtr id;
	// used for LVQ
	int8_t* code_final_ = nullptr;
	// used for adsampling
	INTELLI::TensorPtr transformed = nullptr;
	int level;
	std::vector<std::shared_ptr<HNSWVertex>> neighbors;
	uint8_t visno;
	HNSWVertex(INTELLI::TensorPtr id, int level, int num_neighbors)
			: level(level) {
		this->id = std::move(id);
		visno = 0;
		neighbors =
				std::vector<std::shared_ptr<HNSWVertex>>(num_neighbors, nullptr);
	}
};

typedef std::shared_ptr<HNSWVertex> VertexPtr;
// Table to store visited iteration number during search and insert; Now only
// update the number and store nothing
class VisitedTable {
public:
	// For Tensor t, use visited[TensorPtr] to define if its visited;
	std::unordered_map<INTELLI::TensorPtr, uint8_t> visited_;
	int visno;
	VisitedTable() : visno(1){};
	void set(VertexPtr idx) {
		idx->visno = visno;
		return;
	}
	bool get(VertexPtr idx) { return idx->visno == visno; }

	void set(INTELLI::TensorPtr idx) {
		for (auto it = visited_.begin(); it != visited_.end(); it++) {
			if (torch::equal(*(it->first), *idx)) {
				it->second = visno;
				return;
			}
		}
		visited_[idx] = visno;
	};
	bool get(INTELLI::TensorPtr idx) {
		for (auto it = visited_.begin(); it != visited_.end(); it++) {
			if (torch::equal(*(it->first), *idx)) {
				return it->second == visno;
			}
		}
		return false;
	};
	void advance() {
		if (visno > 250) {
			visno = 0;
			return;
		}
		visno++;
	}
};

#define newVertex make_shared<HNSWVertex>

class HNSW {
public:
	typedef std::pair<float, INTELLI::TensorPtr> Node;
	// sort pairs from nearest to farthest by distance
	struct NodeDistCloser {
		float dist;
		VertexPtr id;
		NodeDistCloser(float dist, VertexPtr id) : dist(dist), id(id){};
		bool operator<(const NodeDistCloser &obj1) const {
			return dist < obj1.dist;
		}
	};
	// sort pairs from farthest to nearest
	struct NodeDistFarther {
		float dist;
		VertexPtr id;
		NodeDistFarther(float dist, VertexPtr id) : dist(dist), id(id){};
		bool operator<(const NodeDistFarther &obj1) const {
			return dist > obj1.dist;
		}
	};
	// a tiny heap that is used during search
	struct MinimaxHeap {
		int n;
		int k;
		int nvalid;

		std::vector<VertexPtr> ids;
		std::vector<float> dis;
		typedef faiss::CMax<float, VertexPtr> HC;
		explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}
		void push(VertexPtr i, float v) {
			if (k == n) {
				if (v >= dis[0]) {
					return;
				}
				if (ids[0] != nullptr) {
					--nvalid;
				}
				faiss::heap_pop<HC>(k--, dis.data(), ids.data());
			}
			faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
			++nvalid;
		};
		float max() const { return dis[0]; };
		int size() const { return nvalid; };
		void clear() {
			nvalid = 0;
			k = 0;
		};
		VertexPtr pop_min(float *vmin_out = nullptr) {
			assert(k > 0);
			// returns min. This is an O(n) operation
			int i = k - 1;
			while (i >= 0) {
				if (ids[i] != nullptr) {
					break;
				}
				i--;
			}
			if (i == -1) {
				return nullptr;
			}
			int imin = i;
			float vmin = dis[i];
			i--;
			while (i >= 0) {
				if (ids[i] != nullptr && dis[i] < vmin) {
					vmin = dis[i];
					imin = i;
				}
				i--;
			}
			if (vmin_out) {
				*vmin_out = vmin;
			}
			auto ret = ids[imin];
			ids[imin] = nullptr;
			--nvalid;
			return ret;
		};
		int count_below(float thresh) {
			int n_below = 0;
			for (int i = 0; i < k; i++) {
				if (dis[i] < thresh) {
					n_below++;
				}
			}
			return n_below;
		}
	};
	// For Tensor t, its assigned levels
	// std::unordered_map<INTELLI::TensorPtr, int> levels_;
	std::vector<int> levels_;
	int64_t vecDim_;
	int64_t ntotal;
	// cumulative number of neighbors stored per layer with that layer excluded,
	// should remain intact! cum_nneighbor_per_level_[0] = 0;
	std::vector<size_t> cum_nneighbor_per_level_;
	// assigned probabilities for each layer (sum=1)
	std::vector<double> probs_of_layers_;
	faiss::RandomGenerator rng;
	// entry point on the top level
	VertexPtr entry_point_ = nullptr;

	typedef int64_t opt_mode_t;
	opt_mode_t opt_mode_ = OPT_VANILLA;
	faiss::MetricType faissMetric = faiss::METRIC_L2;

	// used for LVQ encoding
	std::vector<float> mean_;

	// used for ADsampling
	torch::Tensor transformMatrix;

	// max level of HNSW structure
	size_t max_level_ = -1;
	// entry_point numbers, default as 1
	size_t num_entries = 1;
	// expansion factor at construction time
	size_t efConstruction = 40;
	// expansion factor during search
	size_t efSearch = 15;
	// whether the search process is bounded; now only bounded search is
	// implemented
	bool search_bounded_queue = true;
	// Init HNSW structure with M neighbors
	HNSW(int64_t vecDim, int64_t M) : rng(1919810) {
		vecDim_ = vecDim;
		ntotal = 0;
		set_probs(M, 1 / log(M));
		mean_.resize(vecDim_);
		for (size_t i = 0; i < vecDim_; i++) {
			mean_[i] = 0;
		}
		transformMatrix = AdSampling::getTransformMatrix(vecDim);
	}

	void search(DistanceQueryer &qdis, int k, std::vector<VertexPtr> &I, float *D,
							VisitedTable &vt);
							
	int getLevelsByTensor(torch::Tensor &t);
	
	int getLevelsByPtr(INTELLI::TensorPtr idx);

	void set_nb_neighbors(size_t layer_no, size_t nb);

	size_t nb_neighbors(size_t layer_no);

	size_t cum_nb_neighbors(size_t layer_no);

	int prepare_level_tab(torch::Tensor &x, bool preset_levels, bool is_NSW);

	int random_level();

	void set_probs(int64_t M, float levelMult);

	void neighbor_range(int level, size_t *begin, size_t *end);

	void add_links_starting_from(DistanceQueryer &disq, VertexPtr pt_id,
															 VertexPtr nearest, float d_nearest, int level,
															 VisitedTable &vt);

	void add_without_lock(DistanceQueryer &disq, int assigned_level,
												VertexPtr pt_id, VisitedTable &vt);

	void set_mode(opt_mode_t opt_mode, faiss::MetricType metric);
	string transform_from_tensor(INTELLI::TensorPtr idx);
	HNSW() = default;
};

} 

CANDY::VertexPtr greedy_update_nearest(CANDY::HNSW &hnsw,
		CANDY::DistanceQueryer &disq, int level,
		CANDY::VertexPtr nearest,
		float &d_nearest);

void search_neighbors_to_add(
		CANDY::HNSW &hnsw, CANDY::DistanceQueryer &disq,
		std::priority_queue<CANDY::HNSW::NodeDistCloser> &results,
		CANDY::VertexPtr entry_point, float d_entry_point, int level,
		CANDY::VisitedTable &vt);

void hnsw_shrink_neighbor_list(
		CANDY::DistanceQueryer &disq,
		std::priority_queue<CANDY::HNSW::NodeDistCloser> &resultSet_prev,
		size_t max_size);

void shrink_neighbor_list(
		CANDY::DistanceQueryer &qdis,
		std::priority_queue<CANDY::HNSW::NodeDistFarther> &input,
		std::vector<CANDY::HNSW::NodeDistFarther> &output, size_t max_size);

void add_link(CANDY::HNSW &hnsw, CANDY::DistanceQueryer &disq,
							CANDY::VertexPtr src, CANDY::VertexPtr dest, int level);

std::priority_queue<CANDY::HNSW::Node>
search_from_candidates_unbounded(CANDY::HNSW &hnsw, CANDY::HNSW::Node &node,
																 CANDY::DistanceQueryer &qdis, size_t ef,
																 CANDY::VisitedTable &vt);

int search_from_candidates(CANDY::HNSW &hnsw, CANDY::DistanceQueryer &qdis,
													 int k, std::vector<CANDY::VertexPtr> &I, float *D,
													 CANDY::HNSW::MinimaxHeap &candidates,
													 CANDY::VisitedTable &vt, int level, int nres_in = 0);
#endif 
