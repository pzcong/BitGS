#pragma once
#include<algorithm>
#include<fstream>
#include<cstring>
#include<set>
#include<iostream>
#include <random>
#include"BOBHash.h"
#include<queue>
#define MAX_THREAD 90
typedef uint32_t count_type;
#define sqr(x) ((x)*(x))
#define CAIDA_LEN 13
struct CAIDA_TUPLES {
	uint8_t array[CAIDA_LEN];
};
bool operator == (const CAIDA_TUPLES& a, const CAIDA_TUPLES& b) {
	return memcmp(a.array, b.array, sizeof(CAIDA_TUPLES)) == 0;
}
struct TIMESTAMP {
	uint8_t array[8];
	operator double() { return *(double*)array; }
};
struct EDGE
{
	uint32_t x, y, w;
	EDGE(uint32_t _x, uint32_t _y, uint32_t _w = 0) :x(_x), y(_y), w(_w) {};
};
bool operator < (EDGE x, EDGE y) { return x.w < y.w; }
bool operator > (EDGE x, EDGE y) { return x.w > y.w; }
namespace std {
	template<>
	struct hash<pair<uint32_t, uint32_t>>
	{
		uint32_t operator()(const pair<uint32_t, uint32_t>& item) const noexcept
		{
			return hash32(item, 0);
		}
	};
}
class AdjacentMatrix;
class AdjacentMatrixIterator
{
public:
	AdjacentMatrixIterator(const AdjacentMatrix* ptr, uint32_t _row, uint32_t n)
	{
		obj = ptr;
		row = _row;
		pos = n;
	}
	AdjacentMatrixIterator() :AdjacentMatrixIterator(NULL, 0, 0) {}
	~AdjacentMatrixIterator() {}
	AdjacentMatrixIterator& operator++();
	AdjacentMatrixIterator& operator++(int i);
	AdjacentMatrixIterator& operator--();
	AdjacentMatrixIterator& operator--(int i);
	EDGE operator*() const;
	bool operator!=(const AdjacentMatrixIterator& it)const {
		return it.row != row || it.pos != pos || obj != it.obj;
	}
	bool operator==(const AdjacentMatrixIterator& it)const {
		return it.row == row || it.pos == pos || obj == it.obj;
	}
private:
	const AdjacentMatrix* obj;
	uint32_t row, pos;
};
class AdjacentMatrix
{
	friend class AdjacentMatrixIterator;
public:
	AdjacentMatrix() { total_edge = 0; total_vertex = 0; }
	~AdjacentMatrix() {}

	int check_edge_exist(uint32_t p, uint32_t q) const
	{
		for (uint32_t i = 0; i < edges[p].size(); i++)
		{
			if (edges[p][i].first == q) return 1;
		}
		return 0;
	}
	int get_vertex_size() const
	{
		return total_vertex;
	}
	int get_edge_size() const
	{
		return total_edge;
	}
	uint32_t get_memory_in_byte() const
	{
		return total_vertex * total_vertex;
	}

	void add_edge(uint32_t p, uint32_t q, count_type weight = 1)
	{
		while (total_vertex <= max(p, q))
		{
			total_vertex++;
			edges.push_back(vector<pair<uint32_t, count_type>>());
			stat.push_back(make_pair(0, 0));
		}
		stat[q].first += weight;
		stat[p].second += weight;
		for (uint32_t i = 0; i < edges[p].size(); i++)
		{
			if (edges[p][i].first == q)
			{
				edges[p][i].second += weight;
				return;
			}
		}
		edges[p].push_back(make_pair(q, weight));
		total_edge += 1;
	}
	AdjacentMatrixIterator begin() const
	{
		uint32_t row = 0, col = 0;
		if (edges.size() && edges[row].size() == 0)
			get_next(row, col);
		return AdjacentMatrixIterator(this, row, col);
	}
	AdjacentMatrixIterator end() const
	{
		return AdjacentMatrixIterator(this, total_vertex, 0);
	}
	AdjacentMatrix& operator +=(const AdjacentMatrix& b)
	{
		for (auto e : b)
		{
			add_edge(e.x, e.y, e.w);
		}
		return *this;
	}
	vector<pair<uint32_t, count_type>> get_node_neighbor(uint32_t x) const
	{
		if (x < edges.size())
		{
			auto ret = edges[x];
			sort(ret.begin(), ret.end());
			return ret;
		}
		else
		{
			return vector<pair<uint32_t, count_type>>();
		}
	}
	count_type successor_query(uint32_t x) const
	{
		return stat[x].second;
	}
	count_type precursor_query(uint32_t x) const
	{
		return stat[x].first;
	}
	count_type node_query(uint32_t x) const
	{
		return precursor_query(x) + successor_query(x);
	}
	AdjacentMatrix* shuffle()
	{
		uint32_t mx = 0;
		AdjacentMatrix* ret = new AdjacentMatrix();
		vector<uint32_t> mp(total_vertex);
		for (uint32_t i = 0; i < total_vertex; i++)
		{
			mp[i] = i;
		}
		std::mt19937 gen(3);
		std::shuffle(mp.begin(), mp.end(), gen);
		for (auto e : *this)
		{
			uint32_t x1 = mp[e.x], y1 = mp[e.y];
			ret->add_edge(x1, y1, e.w);
		}
		return ret;
	}
	vector<EDGE> get_edge_topk(uint32_t K) const
	{
		vector<EDGE> ret;
		priority_queue<EDGE, vector<EDGE>, greater<EDGE>> ret1;
		for (auto e : *this)
		{
			if (ret1.size() == K && e < ret1.top())continue;
			if (ret1.size() == K)
				ret1.pop();
			ret1.push(e);
		}
		while (!ret1.empty())
		{
			ret.push_back(ret1.top());
			ret1.pop();
		}
		return ret;
	}
	vector<pair<uint32_t, uint32_t>> get_node_topk(uint32_t K) const
	{
		typedef pair<uint32_t, uint32_t> puu;
		vector<puu> ret;
		priority_queue<puu, vector<puu>, greater<puu>> ret1;
		for (uint32_t i = 0; i < total_vertex; i++)
		{
			if (ret1.size() == K && stat[i].first + stat[i].second <= ret1.top().first) continue;
			if (ret1.size() == K)
			{
				ret1.pop();
			}
			ret1.push(make_pair(stat[i].first + stat[i].second, i));
		}
		while (!ret1.empty())
		{
			ret.push_back(ret1.top());
			ret1.pop();
		}
		return ret;
	}
	static void edge_similarity(const AdjacentMatrix* truth, const AdjacentMatrix* predict,
		double& ARE, double& AAE, double& PR, double& RC, double& F1)
	{
		ARE = AAE = PR = RC = 0;
		double hit = 0;
		int t = min(truth->total_vertex, predict->total_vertex);
		for (uint32_t i = 0; i < t; i++)
		{
			vector<pair<uint32_t, count_type>> a_ls = truth->edges[i], b_ls = predict->edges[i];
			sort(a_ls.begin(), a_ls.end());
			sort(b_ls.begin(), b_ls.end());
			uint32_t idx = 0;
			for (auto& p : a_ls)
			{
				while (idx < b_ls.size() && b_ls[idx].first < p.first)
				{
					AAE += b_ls[idx].second;
					idx++;
				}
				if (idx < b_ls.size() && b_ls[idx].first == p.first)
				{
					if (b_ls[idx].second == p.second)
					{
						hit++;
					}
					ARE += 1.0 * abs(int(b_ls[idx].second - p.second)) / p.second;
					AAE += 1.0 * abs(int(b_ls[idx].second - p.second));
					idx++;
				}
				else
				{
					ARE += 1;
					AAE += p.second;
				}
			}
			while (idx < b_ls.size())
			{
				AAE += b_ls[idx].second;
				idx++;
			}
		}
		for (uint32_t i = t; i < truth->total_vertex; i++)
		{
			for (auto& p : truth->edges[i])
			{
				ARE += 1;
				AAE += p.second;
			}
		}
		ARE /= truth->total_edge;
		AAE /= truth->total_edge;
		PR = hit / predict->get_edge_size();
		RC = hit / truth->get_edge_size();
		F1 = 2 * PR * RC / ((PR + RC) < 1e-9 ? 1 : (PR + RC));
	}
	static void node_similarity(const AdjacentMatrix* truth, vector<uint32_t>* predict,
		double& ARE, double& AAE, double& PR, double& RC, double& F1)
	{
		ARE = AAE = PR = RC = 0;
		double hit = 0, num = 0;
		predict->resize(truth->total_vertex);
		for (uint32_t i = 0; i < min(truth->total_vertex, (uint32_t)predict->size()); i++)
		{
			uint32_t ground_truth = truth->node_query(i), predict_val = (*predict)[i];
			if (ground_truth == predict_val)
			{
				hit++;
			}
			if (predict_val)
			{
				num++;
			}
			ARE += 1.0 * abs(int(predict_val - ground_truth)) / ground_truth;
			AAE += 1.0 * abs(int(predict_val - ground_truth));
		}
		ARE /= truth->total_vertex;
		AAE /= truth->total_vertex;
		PR = hit / num;
		RC = hit / truth->total_vertex;
		F1 = 2 * PR * RC / ((PR + RC) < 1e-9 ? 1 : (PR + RC));
	}
private:
	EDGE get(uint32_t row, uint32_t pos) const
	{
		return EDGE(row, edges[row][pos].first, edges[row][pos].second);
	}
	void get_next(uint32_t& row, uint32_t& pos) const
	{
		pos++;
		while (row < edges.size() && pos >= edges[row].size())
		{
			pos = 0;
			row++;
		}
	}
	void get_last(uint32_t& row, uint32_t& pos) const
	{
		if (pos == 0)
		{
			row--;
		}
		while (row && pos == 0 && edges[row].size() == 0)
		{
			row--;
		}
		if (!row && !pos) return;
		if (pos == 0)
		{
			pos = edges[row].size() - 1;
		}
		else
		{
			pos--;
		}
	}
	vector<vector<pair<uint32_t, count_type>>> edges;
	vector<pair<uint32_t, uint32_t>> stat;
	map<uint32_t, uint32_t> stat1;
	uint32_t total_edge;
	uint32_t total_vertex;
};

AdjacentMatrixIterator& AdjacentMatrixIterator::operator++()
{
	obj->get_next(row, pos);
	return *this;
}
AdjacentMatrixIterator& AdjacentMatrixIterator::operator++(int i)
{
	obj->get_next(row, pos);
	return *this;
}
AdjacentMatrixIterator& AdjacentMatrixIterator::operator--()
{
	obj->get_last(row, pos);
	return *this;
}
AdjacentMatrixIterator& AdjacentMatrixIterator::operator--(int i)
{
	obj->get_last(row, pos);
	return *this;
}
EDGE AdjacentMatrixIterator::operator*() const {
	return obj->get(row, pos);
}