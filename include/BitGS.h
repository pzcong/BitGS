#pragma once
#include"SketchBase.h"
#include"BloomFilter/BloomFilter.h"
#include"BloomFilter/Bitmap.h"
#include<vector>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include<thread>
#ifdef _OPENMP
#include <omp.h>
#endif
#include<queue>
#include<unordered_set>
#include"common/packed_array.h"
#define OFFSET(block_x,block_y,x,y,block_num,block_size) (((block_x) * (block_num) + (block_y)) * sqr(block_size) + (x) * (block_size) + (y))
struct BitGSLayerConfig
{
	uint64_t block_size, fp_len, combine_ratio, block_num, mask;
	BitGSLayerConfig() { block_size = fp_len = combine_ratio = block_num = 0; }
	BitGSLayerConfig(uint64_t _block_len, uint64_t _fp_len, uint64_t _combine_ratio) :block_size(_block_len), fp_len(_fp_len), combine_ratio(_combine_ratio), block_num(0) { mask = (1 << fp_len) - 1; };
};
inline uint64_t cal_hash_num(uint64_t memory_in_byte, uint64_t n)
{
	return max(1.0, memory_in_byte * 8 * 0.7 / n + 0.5);
}
class BitGS :public SketchBase
{
public:
	BitGS(uint64_t memory_in_byte, uint64_t _hsh_num, uint64_t max_idx, uint64_t edge_size, vector<BitGSLayerConfig> _layer_config
		, uint64_t bf_mem = 0, uint64_t pruning_mem = 0, uint64_t pruning1_mem = 0, bool use_bitmap = false, string suffix = "")
		:SketchBase("BitGS" + ((suffix != "") ? "_" + suffix : ""))
	{
		hsh_num = _hsh_num;
		memory_in_byte = max(memory_in_byte, 100000u + bf_mem + pruning_mem + pruning1_mem) - bf_mem - pruning_mem - pruning1_mem;
		uint64_t block_num = cal_first_layer_block_num(memory_in_byte, _layer_config);
		BitGSLayerConfig init(1, 0, ceil(1.0 * max_idx / block_num));
		init.block_num = max_idx;
		layer_config = vector<BitGSLayerConfig>(_layer_config.size() + 1);
		layer_config[0] = init;
		copy(_layer_config.begin(), _layer_config.end(), layer_config.begin() + 1);
		LAYERS = layer_config.size();
		A = new uint32_t * [LAYERS];
		//stat = new uint64_t * [LAYERS];
		overflow = new BloomFilterBase * [LAYERS];
		for (uint64_t i = 1; i < LAYERS; i++)
		{
			layer_config[i].block_num = block_num;
			uint64_t total_len = ceil(1.0 * sqr(block_num) * sqr(layer_config[i].block_size) * layer_config[i].fp_len / 8);
			A[i] = new uint32_t[total_len / 4];
			//stat[i] = new uint64_t[sqr(block_num) * sqr(layer_config[i].block_size)];
			//memset(stat[i], 0, sizeof(uint64_t) * sqr(block_num) * sqr(layer_config[i].block_size));
			memset(A[i], 0, total_len);
			overflow[i] = new Bitmap(block_num * layer_config[i].block_size);
			block_num = ceil(1.0 * block_num / layer_config[i].combine_ratio);
		}
		if (use_bitmap)
		{
			overflow[0] = new Bitmap(max_idx);
		}
		else if (bf_mem)
		{
			overflow[0] = new BloomFilter(bf_mem, cal_hash_num(bf_mem, edge_size));
		}
		else
		{
			overflow[0] = NULL;
		}
		if (pruning_mem)
		{
			pruning = new BloomFilter(pruning_mem, cal_hash_num(pruning_mem, sqrt((double)max_idx * (double)edge_size)));
		}
		else
		{
			pruning = NULL;
		}
		if (pruning1_mem)
		{
			pruning1 = new BloomFilter(pruning_mem, cal_hash_num(pruning_mem, sqrt((double)max_idx * (double)edge_size)));
		}
		else
		{
			pruning1 = NULL;
		}
	}
	~BitGS()
	{
		for (uint64_t i = 1; i < LAYERS; i++)
		{
			delete[] A[i];
			A[i] = NULL;
			//delete[] stat[i];
			//stat[i] = NULL;
			delete overflow[i];
			overflow[i] = NULL;
		}
		delete[] A;
		A = NULL;
		if (overflow[0] != NULL)
			delete overflow[0];
		delete overflow;
		overflow = NULL;
		//delete[] stat;
		//stat = NULL;
		if (pruning != NULL)
		{
			delete pruning;
			pruning = NULL;
		}
		if (pruning1 != NULL)
		{
			delete pruning1;
			pruning1 = NULL;
		}
	}
	virtual uint32_t get_memory_in_byte() const
	{
		return 1;
	}
	virtual int insert(uint32_t x, uint32_t y, uint32_t w)
	{
		start_timing();
		insert(0, x, y, 0, 0, w);
		insert_time += stop_timing();
		return 1;
	}
	virtual count_type edge_query(uint32_t x, uint32_t y)
	{
		start_timing();
		count_type ret = 0;
		ret = query(0, x, y, 0, 0);
		edge_query_time += stop_timing();
		return ret;
	}
	virtual count_type successor_query(uint32_t x)
	{
		uint64_t block_x = x / layer_config[0].combine_ratio;
		uint64_t ret = 0, tmp = 0;
		uint64_t* src_x = new uint64_t[hsh_num];
		for (uint64_t i = 0; i < hsh_num; i++)
		{
			src_x[i] = hash32(x, i) % layer_config[1].block_size;
		}
		for (uint64_t block_y = 0; block_y < layer_config[1].block_num; block_y++)
		{
			if (pruning != NULL && pruning->query(x, block_y) == 0) continue;
			query_one_block(&tmp, block_x, block_y, src_x, 0);
			ret += tmp;
		}
		delete[] src_x;
		return ret;
	}
	virtual count_type precursor_query(uint32_t y)
	{
		uint64_t block_y = y / layer_config[0].combine_ratio;
		uint64_t ret = 0, tmp = 0;
		uint64_t* src_y = new uint64_t[hsh_num];
		for (uint64_t i = 0; i < hsh_num; i++)
		{
			src_y[i] = hash32(y, i + hsh_num) % layer_config[1].block_size;
		}
		for (uint64_t block_x = 0; block_x < layer_config[1].block_num; block_x++)
		{
			if (pruning1 != NULL && pruning1->query(y, block_x) == 0) continue;
			query_one_block(&tmp, block_y, block_x, src_y, 1);
			ret += tmp;
		}
		delete[] src_y;
		return ret;
	}
	virtual double edgeDecode(AdjacentMatrix*& ret)
	{
		decode_time1 = decode_time2 = decode_time3 = 0;
		start_timing();
		uint64_t total_length = sqr(layer_config[LAYERS - 1].block_num) * sqr(layer_config[LAYERS - 1].block_size);
		uint64_t* last_layer_recover = new uint64_t[total_length];
		for (uint64_t i = 0; i < total_length; i++)
		{
			last_layer_recover[i] = ACCESS32(A[LAYERS - 1], layer_config[LAYERS - 1].fp_len, i);
		}
		for (int l = LAYERS - 1; l >= 1; l--)
		{
			total_length = sqr(layer_config[l - 1].block_num) * sqr(layer_config[l - 1].block_size);
			vector<vector<EDGE>*> recovers;
			vector<pair<thread, bool*>> ths;
			for (uint64_t x = 0; x < layer_config[l].block_num; x++)
			{
				for (uint64_t y = 0; y < layer_config[l].block_num; y++)
				{
					if (ths.size() == MAX_THREAD)
					{
						auto selectItem = ths.end();
						while (selectItem == ths.end())
						{
							for (auto p = ths.begin(); p < ths.end(); p++)
							{
								if (*(p->second) == true)
								{
									selectItem = p;
									break;
								}
							}
						}
						selectItem->first.join();
						delete selectItem->second;
						ths.erase(selectItem);
						//cout << ths.size() << endl;
					}
					recovers.push_back(new vector<EDGE>());
					bool* reply = new bool(false);
					ths.push_back(make_pair(thread(&BitGS::edge_decode_one_block, this, recovers[recovers.size() - 1], reply, last_layer_recover, (x * layer_config[l].block_num + y) * sqr(layer_config[l].block_size), layer_config[l].block_size, layer_config[l].block_num
						, l - 1, x * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, y * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size), reply));
					/*edge_decode_one_block(recovers[recovers.size() - 1], reply, last_layer_recover, (x * layer_config[l].block_num + y) * sqr(layer_config[l].block_size), layer_config[l].block_size, layer_config[l].block_num
						, l - 1, x * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, y * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size);*/
				}
			}
			for (auto& p : ths)
			{
				p.first.join();
				delete p.second;
			}
			ths.clear();
			if (l != 1)
			{
				delete[] last_layer_recover;
				total_length = sqr(layer_config[l - 1].block_num) * sqr(layer_config[l - 1].block_size);
				last_layer_recover = new uint64_t[total_length];
				for (uint64_t i = 0; i < total_length; i++)
				{
					last_layer_recover[i] = ACCESS32(A[l - 1], layer_config[l - 1].fp_len, i);
				}
				int cnt = 0;
				for (auto& recover : recovers)
				{
					cnt += recover->size();
					for (auto& e : *recover)
					{
						last_layer_recover[OFFSET(e.x / layer_config[l - 1].block_size, e.y / layer_config[l - 1].block_size, e.x % layer_config[l - 1].block_size, e.y % layer_config[l - 1].block_size, layer_config[l - 1].block_num, layer_config[l - 1].block_size)] += e.w << layer_config[l - 1].fp_len;
					}
					delete recover;
				}
			}
			else
			{
				delete[] last_layer_recover;
				ret = new AdjacentMatrix();
				int cnt = 0;
				for (auto& recover : recovers)
				{
					cnt += recover->size();
					for (auto& e : *recover)
					{
						ret->add_edge(e.x, e.y, e.w);
					}
					delete recover;
				}
			}
		}
		decode_time = stop_timing();
		//cout << "scan:" << decode_time1 / 1e9 << "s elimination:" << decode_time2 / 1e9 << " Eigen:" << decode_time3 / 1e9 << "s\n";
		return 0.0;
	}
	virtual double nodeDecode(vector<uint32_t>*& ret)
	{
		decode_time1 = decode_time2 = decode_time3 = 0;
		start_timing();
		uint64_t total_length = sqr(layer_config[LAYERS - 1].block_num) * sqr(layer_config[LAYERS - 1].block_size);
		uint64_t* last_layer_recover = new uint64_t[total_length];
		for (uint64_t i = 0; i < total_length; i++)
		{
			last_layer_recover[i] = ACCESS32(A[LAYERS - 1], layer_config[LAYERS - 1].fp_len, i);
		}
		for (int l = LAYERS - 1; l >= 1; l--)
		{
			total_length = sqr(layer_config[l - 1].block_num) * sqr(layer_config[l - 1].block_size);
			vector<vector<EDGE>*> recovers;
			vector<pair<thread, bool*>> ths;
			for (uint64_t x = 0; x < layer_config[l].block_num; x++)
			{
				for (uint64_t y = 0; y < layer_config[l].block_num; y++)
				{
					if (ths.size() == MAX_THREAD)
					{
						auto selectItem = ths.end();
						while (selectItem == ths.end())
						{
							for (auto p = ths.begin(); p < ths.end(); p++)
							{
								if (*(p->second) == true)
								{
									selectItem = p;
									break;
								}
							}
						}
						selectItem->first.join();
						delete selectItem->second;
						ths.erase(selectItem);
						//ths = queue<pair<thread, bool*>>(ths1);
					}
					recovers.push_back(new vector<EDGE>());
					bool* reply = new bool(false);
					if (l != 1)
					{
						ths.push_back(make_pair(thread(&BitGS::edge_decode_one_block, this, recovers[recovers.size() - 1], reply, last_layer_recover, (x * layer_config[l].block_num + y) * sqr(layer_config[l].block_size), layer_config[l].block_size, layer_config[l].block_num
							, l - 1, x * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, y * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size), reply));
					}
					else
					{
						//ths.push_back(make_pair(thread(&BitGS::node_decode_one_block, this, recovers[recovers.size() - 1], reply, last_layer_recover, (x * layer_config[l].block_num + y) * sqr(layer_config[l].block_size), layer_config[l].block_size, layer_config[l].block_num
						//	, l - 1, x * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, y * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size), reply));
						node_decode_one_block(recovers[recovers.size() - 1], reply, last_layer_recover, (x * layer_config[l].block_num + y) * sqr(layer_config[l].block_size), layer_config[l].block_size, layer_config[l].block_num
							, l - 1, x * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, y * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size);
					}
					/*edge_decode_one_block(recovers[recovers.size() - 1], reply, last_layer_recover, (x * layer_config[l].block_num + y) * sqr(layer_config[l].block_size), layer_config[l].block_size, layer_config[l].block_num
						, l - 1, x * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, y * layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size, layer_config[l - 1].combine_ratio * layer_config[l - 1].block_size);*/
				}
			}
			for (auto& p : ths)
			{
				p.first.join();
				delete p.second;
			}
			ths.clear();
			if (l != 1)
			{
				delete[] last_layer_recover;
				total_length = sqr(layer_config[l - 1].block_num) * sqr(layer_config[l - 1].block_size);
				last_layer_recover = new uint64_t[total_length];
				for (uint64_t i = 0; i < total_length; i++)
				{
					last_layer_recover[i] = ACCESS32(A[l - 1], layer_config[l - 1].fp_len, i);
				}
				for (auto& recover : recovers)
				{
					for (auto& e : *recover)
					{
						last_layer_recover[OFFSET(e.x / layer_config[l - 1].block_size, e.y / layer_config[l - 1].block_size, e.x % layer_config[l - 1].block_size, e.y % layer_config[l - 1].block_size, layer_config[l - 1].block_num, layer_config[l - 1].block_size)] += e.w << layer_config[l - 1].fp_len;
					}
					delete recover;
				}
			}
			else
			{
				delete[] last_layer_recover;
				map<uint32_t, uint32_t> dict;
				uint32_t mx_id = 0;
				for (auto& recover : recovers)
				{
					for (auto& e : *recover)
					{
						dict[e.x] += e.w;
						mx_id = max(mx_id, e.x);
					}
					delete recover;
				}
				ret = new vector<uint32_t>(mx_id + 1);
				for (auto& p : dict)
				{
					(*ret)[p.first] = p.second;
				}
			}
		}
		decode_time = stop_timing();
		//cout << "scan:" << decode_time1 / 1e9 << "s elimination:" << decode_time2 / 1e9 << " Eigen:" << decode_time3 / 1e9 << "s\n";
		return 0.0;
	}
	uint64_t decode_time1 = 0, decode_time2 = 0, decode_time3 = 0;
private:
	void insert(uint64_t src_layer, uint64_t src_block_x, uint64_t src_block_y, uint64_t src_x, uint64_t src_y, uint64_t w)
	{
		uint64_t combine_ratio = layer_config[src_layer].combine_ratio;
		uint64_t target_block_x = src_block_x / combine_ratio, target_block_y = src_block_y / combine_ratio;
		uint64_t target_block_offset = (target_block_x * layer_config[src_layer + 1].block_num + target_block_y) * sqr(layer_config[src_layer + 1].block_size);
		uint64_t src_offset = OFFSET(src_block_x, src_block_y, src_x, src_y, layer_config[src_layer].block_num, layer_config[src_layer].block_size);
		uint64_t src_x_offset = src_block_x * layer_config[src_layer].block_size + src_x;
		uint64_t src_y_offset = src_block_y * layer_config[src_layer].block_size + src_y;
		if (src_layer || overflow[0] != NULL)
			overflow[src_layer]->insert(src_x_offset, src_y_offset);
		if (src_layer == 0 && pruning != NULL)
		{
			pruning->insert(src_x_offset, target_block_y);
		}
		if (src_layer == 0 && pruning1 != NULL)
		{
			pruning1->insert(src_y_offset, target_block_x);
		}
		pair<uint64_t, uint64_t> pr(src_x_offset, src_y_offset);
		uint64_t block_square = sqr(layer_config[src_layer + 1].block_size);
		for (uint64_t j = 0; j < hsh_num; j++)
		{
			uint64_t hash_code;
			if (src_layer)
			{
				hash_code = hash32(pr, j + hsh_num * src_layer) % block_square;
			}
			else
			{
				uint64_t target_x = hash32(src_x_offset, j + hsh_num * src_layer * 2) % layer_config[src_layer + 1].block_size,
					target_y = hash32(src_y_offset, j + hsh_num * src_layer * 2 + hsh_num) % layer_config[src_layer + 1].block_size;
				hash_code = target_x * layer_config[src_layer + 1].block_size + target_y;
			}
			uint64_t offset = target_block_offset + hash_code;
			uint64_t val = ACCESS32(A[src_layer + 1], layer_config[src_layer + 1].fp_len, offset);
			//stat[src_layer + 1][offset] += w;
			if (val + w <= layer_config[src_layer + 1].mask)
			{
				val += w;
				WRITE32(A[src_layer + 1], layer_config[src_layer + 1].fp_len, offset, val);
			}
			else if (src_layer != LAYERS - 2)
			{
				uint64_t next_w = (val + w) >> layer_config[src_layer + 1].fp_len;
				val = (val + w) & layer_config[src_layer + 1].mask;
				WRITE32(A[src_layer + 1], layer_config[src_layer + 1].fp_len, offset, val);
				insert(src_layer + 1, target_block_x, target_block_y, hash_code / layer_config[src_layer + 1].block_size, hash_code % layer_config[src_layer + 1].block_size, next_w);
			}
			else
			{
				WRITE32(A[src_layer + 1], layer_config[src_layer + 1].fp_len, offset, layer_config[src_layer + 1].mask);
			}
		}
	}
	count_type query(uint64_t src_layer, uint64_t src_block_x, uint64_t src_block_y, uint64_t src_x, uint64_t src_y, uint64_t optimal = 0xFFFFFFFF)
	{
		uint64_t combine_ratio = layer_config[src_layer].combine_ratio;
		uint64_t src_offset = OFFSET(src_block_x, src_block_y, src_x, src_y, layer_config[src_layer].block_num, layer_config[src_layer].block_size);
		uint64_t src_x_offset = src_block_x * layer_config[src_layer].block_size + src_x;
		uint64_t src_y_offset = src_block_y * layer_config[src_layer].block_size + src_y;
		count_type overflows = 0xFFFFFFFF, init = 0;
		if (src_layer != 0)
		{
			init = ACCESS32(A[src_layer], layer_config[src_layer].fp_len, src_offset);
		}
		if (src_layer == LAYERS - 1 || init >= optimal || ((src_layer || overflow[0] != NULL) && !overflow[src_layer]->query(src_x_offset, src_y_offset)))
		{
			return init;
		}
		else
		{
			if (init + layer_config[src_layer].mask >= optimal)
			{
				return init + layer_config[src_layer].mask;
			}
			uint64_t target_block_x = src_block_x / combine_ratio, target_block_y = src_block_y / combine_ratio, block_square = sqr(layer_config[src_layer + 1].block_size);
			uint64_t target_block_offset = (target_block_x * layer_config[src_layer + 1].block_num + target_block_y) * block_square;
			pair<uint64_t, uint64_t> pr(src_x_offset, src_y_offset);
			for (uint64_t j = 0; j < hsh_num; j++)
			{
				uint64_t hash_code;
				if (src_layer)
				{
					hash_code = hash32(pr, j + hsh_num * src_layer) % block_square;
				}
				else
				{
					uint64_t target_x = hash32(src_x_offset, j + hsh_num * src_layer * 2) % layer_config[src_layer + 1].block_size,
						target_y = hash32(src_y_offset, j + hsh_num * src_layer * 2 + hsh_num) % layer_config[src_layer + 1].block_size;
					hash_code = target_x * layer_config[src_layer + 1].block_size + target_y;
				}
				uint64_t offset = target_block_offset + hash_code;
				overflows = min(overflows, query(src_layer + 1, target_block_x, target_block_y,
					hash_code / layer_config[src_layer + 1].block_size, hash_code % layer_config[src_layer + 1].block_size, overflows));
			}
			return (overflows << layer_config[src_layer].fp_len) + init;
		}
	}
	void edge_decode_one_block(vector<EDGE>* src, bool* reply, const uint64_t* target, uint64_t target_block_offset, uint64_t target_block_size, uint64_t target_block_num
		, uint64_t src_layer, uint64_t src_x_start, uint64_t src_y_start, uint64_t src_block_size)
	{
		//A*src=target
#ifdef WIN32

		LARGE_INTEGER t1;
#else
		timespec t1;
#endif
		start_timing(t1);
		vector<Eigen::Triplet<double>> triplet_list;
		unordered_set<pair<uint32_t, uint32_t>> nonzero_list;
		map<uint64_t, map<uint64_t, uint64_t>>mp;
		uint64_t cnt = 0;
		uint64_t y_end = min(src_y_start + src_block_size, layer_config[src_layer].block_size * layer_config[src_layer].block_num),
			x_end = min(src_x_start + src_block_size, layer_config[src_layer].block_size * layer_config[src_layer].block_num);
		vector<uint64_t> alt_x, alt_y;
		for (uint64_t x = src_x_start; x < x_end; x++)
		{
			if (!src_layer && pruning != NULL && !pruning->query(x, src_y_start / src_block_size)) continue;
			alt_x.push_back(x);
		}
		for (uint64_t y = src_y_start; y < y_end; y++)
		{
			if (!src_layer && pruning1 != NULL && !pruning1->query(y, src_x_start / src_block_size)) continue;
			alt_y.push_back(y);
		}
		for (auto& x : alt_x)
		{
			for (auto& y : alt_y)
			{
				if (!overflow[src_layer]->query(x, y)) continue;
				nonzero_list.insert(make_pair(x, y));
				pair<uint64_t, uint64_t> pr(x, y);
				for (uint64_t i = 0; i < hsh_num; i++)
				{
					uint64_t hash_code;
					if (src_layer)
					{
						hash_code = hash32(pr, src_layer * hsh_num + i) % sqr(target_block_size);
					}
					else
					{
						uint64_t target_x = hash32(x, i + hsh_num * src_layer * 2) % target_block_size,
							target_y = hash32(y, i + hsh_num * src_layer * 2 + hsh_num) % target_block_size;
						hash_code = target_x * target_block_size + target_y;
					}
					mp[hash_code][(x - src_x_start) * src_block_size + y - src_y_start]++;
				}
			}
		}
		decode_time1 += stop_timing(t1);
		start_timing(t1);
		double* b1 = new double[sqr(target_block_size)];
		copy(target + target_block_offset, target + (target_block_offset + sqr(target_block_size)), b1);
		queue<uint64_t> lst;
		vector<bool> visited(sqr(target_block_size), 0);
		for (const auto& p : mp)
		{
			if (p.second.size() == 1 || b1[p.first] == 0)
			{
				lst.push(p.first);
				visited[p.first] = true;
			}
		}
		while (!lst.empty())
		{
			uint64_t target_idx = lst.front();
			lst.pop();
			for (auto& pr : mp[target_idx])
			{
				uint64_t src_idx = pr.first, cnt = b1[target_idx] / pr.second;
				uint64_t x = src_idx / src_block_size + src_x_start, y = src_idx % src_block_size + src_y_start;
				pair<uint64_t, uint64_t> pr_src(x, y);
				map<uint64_t, uint64_t> tmp;
				for (uint64_t i = 0; i < hsh_num; i++)
				{
					uint64_t remove_target_idx;
					if (src_layer)
					{
						remove_target_idx = hash32(pr_src, src_layer * hsh_num + i) % sqr(target_block_size);
					}
					else
					{
						uint64_t target_x = hash32(x, i + hsh_num * src_layer * 2) % target_block_size,
							target_y = hash32(y, i + hsh_num * src_layer * 2 + hsh_num) % target_block_size;
						remove_target_idx = target_x * target_block_size + target_y;
					}
					tmp[remove_target_idx]++;
					cnt = min(cnt, (uint64_t)b1[remove_target_idx] / tmp[remove_target_idx]);
				}
				if (nonzero_list.find(make_pair(x, y)) != nonzero_list.end())
				{
					nonzero_list.erase(make_pair(x, y));
				}
				else
				{
					std::cout << "error:" << src_layer << "," << (src_x_start / src_block_size == 0) << "," << (src_y_start / src_block_size == 6) << "," << target_idx << "," << x << "," << y << endl;
				}
				if (cnt)
					src->push_back(EDGE(x, y, cnt));
				for (uint64_t i = 0; i < hsh_num; i++)
				{
					uint64_t remove_target_idx;
					if (src_layer)
					{
						remove_target_idx = hash32(pr_src, src_layer * hsh_num + i) % sqr(target_block_size);
					}
					else
					{
						uint64_t target_x = hash32(x, i + hsh_num * src_layer * 2) % target_block_size,
							target_y = hash32(y, i + hsh_num * src_layer * 2 + hsh_num) % target_block_size;
						remove_target_idx = target_x * target_block_size + target_y;
					}

					mp[remove_target_idx][src_idx]--;
					b1[remove_target_idx] -= cnt;
					if (mp[remove_target_idx][src_idx] == 0 && remove_target_idx != target_idx)
					{
						mp[remove_target_idx].erase(src_idx);
						if (mp[remove_target_idx].size() == 1 && visited[remove_target_idx] == false)
						{
							lst.push(remove_target_idx);
							visited[remove_target_idx] == true;
						}
					}
					if (b1[remove_target_idx] == 0 && visited[remove_target_idx] == false)
					{
						lst.push(remove_target_idx);
						visited[remove_target_idx] == true;
					}
				}
			}
			mp.erase(target_idx);
		}
		decode_time2 += stop_timing(t1);
		start_timing(t1);
		if (mp.size())
		{
			map<uint64_t, uint64_t> dict1;
			uint64_t cnt = 0;
			for (auto i : nonzero_list)
			{
				uint64_t x = i.first, y = i.second;
				dict1[(x - src_x_start) * src_block_size + y - src_y_start] = cnt++;
			}

			for (auto& p : mp)
			{
				for (auto& q : p.second)
				{
					triplet_list.push_back(Eigen::Triplet<double>(p.first, dict1[q.first], q.second));
				}
			}
			Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>>solver_sparse;
			Eigen::SparseMatrix<double> A(sqr(target_block_size), nonzero_list.size());
			A.setFromTriplets(triplet_list.begin(), triplet_list.end());
			A.makeCompressed();
			//solver_sparse.setMaxIterations(1);
			solver_sparse.compute(A);

			Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(b1, sqr(target_block_size));
			Eigen::VectorXd restore = solver_sparse.solve(b);
			for (auto p : nonzero_list)
			{
				uint64_t x = p.first, y = p.second, idx = (x - src_x_start) * src_block_size + y - src_y_start;
				double t = restore[dict1[idx]];
				if (t >= 0.5)
					src->push_back(EDGE(x, y, t + 0.5));
			}
		}
		decode_time3 += stop_timing(t1);
		delete[] b1;
		b1 = NULL;
		*reply = true;
	}
	void node_decode_one_block(vector<EDGE>* src, bool* reply, const uint64_t* target, uint64_t target_block_offset, uint64_t target_block_size, uint64_t target_block_num
		, uint64_t src_layer, uint64_t src_x_start, uint64_t src_y_start, uint64_t src_block_size)
	{
		//A*src=target
#ifdef WIN32

		LARGE_INTEGER t1;
#else
		timespec t1;
#endif
		start_timing(t1);
		unordered_set<uint32_t> nonzero_list_x, nonzero_list_y;
		map<uint64_t, map<uint64_t, uint64_t>>mp_x, mp_y;
		uint64_t cnt = 0;
		uint64_t y_end = min(src_y_start + src_block_size, layer_config[src_layer].block_size * layer_config[src_layer].block_num),
			x_end = min(src_x_start + src_block_size, layer_config[src_layer].block_size * layer_config[src_layer].block_num);
		vector<uint64_t> alt_x, alt_y;
		for (uint64_t x = src_x_start; x < x_end; x++)
		{
			if (!src_layer && pruning != NULL && !pruning->query(x, src_y_start / src_block_size)) continue;
			alt_x.push_back(x);
		}

		for (auto& x : alt_x)
		{
			nonzero_list_x.insert(x);
			for (uint64_t i = 0; i < hsh_num; i++)
			{
				uint64_t target_x = hash32(x, i) % target_block_size;
				mp_x[target_x][x - src_x_start]++;
			}
		}
		for (uint64_t y = src_y_start; y < y_end; y++)
		{
			if (!src_layer && pruning1 != NULL && !pruning1->query(y, src_x_start / src_block_size)) continue;
			alt_y.push_back(y);
		}
		for (auto& y : alt_y)
		{
			nonzero_list_y.insert(y);
			for (uint64_t i = 0; i < hsh_num; i++)
			{
				uint64_t target_y = hash32(y, i + hsh_num) % target_block_size;
				mp_y[target_y][y - src_y_start]++;
			}
		}
		decode_time1 += stop_timing(t1);
		start_timing(t1);
		double* b1 = new double[target_block_size], * b2 = new double[target_block_size];
		memset(b1, 0, target_block_size * sizeof(double));
		memset(b2, 0, target_block_size * sizeof(double));
		for (size_t i = 0; i < target_block_size; i++)
		{
			for (size_t j = 0; j < target_block_size; j++)
			{
				b1[i] += target[target_block_offset + i * target_block_size + j];
				b2[j] += target[target_block_offset + i * target_block_size + j];
			}
		}
		elimination_of_unknown(0, target_block_size, src_x_start, src_block_size, nonzero_list_x, mp_x, src, b1);
		elimination_of_unknown(1, target_block_size, src_y_start, src_block_size, nonzero_list_y, mp_y, src, b2);
		decode_time2 += stop_timing(t1);
		start_timing(t1);
		if (nonzero_list_x.find(0) != nonzero_list_x.end())
		{
			int t = 1;
		}
		if (mp_x.size())
		{
			lsqr(0, target_block_size, src_x_start, src_block_size, nonzero_list_x, mp_x, src, b1);
		}
		if (mp_y.size())
		{
			lsqr(1, target_block_size, src_y_start, src_block_size, nonzero_list_y, mp_y, src, b2);
		}
		decode_time3 += stop_timing(t1);
		delete[] b1;
		b1 = NULL;
		*reply = true;
	}
	static uint64_t cal_first_layer_block_num(uint64_t memory, vector<BitGSLayerConfig> layer_config)
	{
		uint64_t LAYERS = layer_config.size(), l = 0, r = sqrt(memory * 8 / layer_config[0].fp_len) / layer_config[0].block_size, mid;
		while (l < r)
		{
			mid = (l + r + 1) / 2;
			uint64_t mem_need = 0, block_num = mid;
			for (int i = 0; i < LAYERS && mem_need / 8 <= memory; i++)
			{
				mem_need += sqr(block_num * layer_config[i].block_size) * layer_config[i].fp_len;
				block_num = ceil(1.0 * block_num / layer_config[i].combine_ratio);
			}
			if (mem_need / 8 > memory)
			{
				r = mid - 1;
			}
			else
			{
				l = mid;
			}
		}
		return r;
	}
	void query_one_block(uint64_t* ret, uint64_t block_x, uint64_t block_y, uint64_t* src_x, uint64_t dim)
	{
		*ret = 0xFFFFFFFF;
		uint64_t sum;
		if (dim == 0)
		{
			for (uint64_t i = 0; i < hsh_num; i++)
			{
				sum = 0;
				for (uint64_t src_y = 0; src_y < layer_config[1].block_size && sum < *ret; src_y++)
				{
					sum += query(1, block_x, block_y, src_x[i], src_y);
				}
				*ret = min(*ret, sum);
			}
		}
		else
		{
			for (uint64_t i = 0; i < hsh_num; i++)
			{
				sum = 0;
				for (uint64_t src_y = 0; src_y < layer_config[1].block_size && sum < *ret; src_y++)
				{
					sum += query(1, block_y, block_x, src_y, src_x[i]);
				}
				*ret = min(*ret, sum);
			}
		}
	}
	void elimination_of_unknown(uint64_t dim, uint64_t target_block_size, uint64_t src_start, uint64_t src_block_size,
		unordered_set<uint32_t>& nonzero_list, map<uint64_t, map<uint64_t, uint64_t>>& mp, vector<EDGE>* src, double* b)
	{
		queue<uint64_t> lst;
		vector<bool> visited(target_block_size, 0);
		for (const auto& p : mp)
		{
			if (p.second.size() == 1 || b[p.first] == 0)
			{
				lst.push(p.first);
				visited[p.first] = true;
			}
		}
		while (!lst.empty())
		{
			uint64_t target_idx = lst.front();
			lst.pop();
			for (auto& pr : mp[target_idx])
			{
				uint64_t src_idx = pr.first, cnt = b[target_idx] / pr.second;
				uint64_t x = src_idx + src_start;
				map<uint64_t, uint64_t> tmp;
				for (uint64_t i = 0; i < hsh_num; i++)
				{
					uint64_t remove_target_idx = hash32(x, i + hsh_num * dim) % target_block_size;
					tmp[remove_target_idx]++;
					cnt = min(cnt, (uint64_t)b[remove_target_idx] / tmp[remove_target_idx]);
				}
				nonzero_list.erase(x);
				if (cnt)
					src->push_back(EDGE(x, dim, cnt));
				for (uint64_t i = 0; i < hsh_num; i++)
				{
					uint64_t remove_target_idx = hash32(x, i + hsh_num * dim) % target_block_size;

					mp[remove_target_idx][src_idx]--;
					b[remove_target_idx] -= cnt;
					if (mp[remove_target_idx][src_idx] == 0 && remove_target_idx != target_idx)
					{
						mp[remove_target_idx].erase(src_idx);
						if (mp[remove_target_idx].size() == 1 && visited[remove_target_idx] == false)
						{
							lst.push(remove_target_idx);
							visited[remove_target_idx] == true;
						}
					}
					if (b[remove_target_idx] == 0 && visited[remove_target_idx] == false)
					{
						lst.push(remove_target_idx);
						visited[remove_target_idx] == true;
					}
				}
			}
			mp.erase(target_idx);
		}
	}
	void lsqr(uint64_t dim, uint64_t target_block_size, uint64_t src_start, uint64_t src_block_size,
		unordered_set<uint32_t>& nonzero_list, map<uint64_t, map<uint64_t, uint64_t>>& mp, vector<EDGE>* src, double* b1)
	{
		vector<Eigen::Triplet<double>> triplet_list;
		map<uint64_t, uint64_t> dict1;
		uint64_t cnt = 0;
		for (auto i : nonzero_list)
		{
			dict1[i - src_start] = cnt++;
		}

		for (auto& p : mp)
		{
			for (auto& q : p.second)
			{
				triplet_list.push_back(Eigen::Triplet<double>(p.first, dict1[q.first], q.second));
			}
		}
		Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>>solver_sparse;
		Eigen::SparseMatrix<double> A(target_block_size, nonzero_list.size());
		A.setFromTriplets(triplet_list.begin(), triplet_list.end());
		A.makeCompressed();
		//solver_sparse.setMaxIterations(1);
		solver_sparse.compute(A);

		Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(b1, target_block_size);
		Eigen::VectorXd restore = solver_sparse.solve(b);
		for (auto p : nonzero_list)
		{
			uint64_t idx = p - src_start;
			double t = restore[dict1[idx]];
			if (t >= 0.5)
			{
				if (p == 0)
				{
					int t = 1;
				}
				src->push_back(EDGE(p, dim, t + 0.5));
			}
		}
	}
	uint64_t num, hsh_num, LAYERS;
	uint32_t** A;
	//uint64_t** stat;
	BloomFilterBase** overflow;
	BloomFilter* pruning, * pruning1;
	vector<BitGSLayerConfig> layer_config;
};
