#pragma once
#include<string>
#include <chrono>
#include<time.h>
#ifdef WIN32
#include <windows.h>
#endif
#include"common/util.h"
class SketchBase
{
public:
	SketchBase(string _name) :name(_name) { insert_time = edge_query_time = node_query_time = 0; }

	virtual ~SketchBase() {}
	string get_name() const { return name; }
	void compress(const AdjacentMatrix* adj)
	{
		int cnt = 0;
		for (auto e : *adj)
		{
			//cout << e.x << " " << e.y << " " << e.w << endl;
			cnt++;
			insert(e.x, e.y, e.w);
		}
	}
	virtual uint32_t get_memory_in_byte() const = 0;
	virtual int insert(uint32_t x, uint32_t y, uint32_t w) = 0;
	virtual count_type edge_query(uint32_t x, uint32_t y) = 0;
	virtual count_type precursor_query(uint32_t x) = 0;
	virtual count_type successor_query(uint32_t x) = 0;
	virtual count_type node_query(uint32_t x)
	{
		start_timing();
		auto ret = precursor_query(x) + successor_query(x);
		node_query_time += stop_timing();
		return ret;
	}
	virtual double edgeDecode(AdjacentMatrix*& ret)
	{
		ret = new AdjacentMatrix();
		return 0.0;
	};
	virtual double nodeDecode(vector<uint32_t>*& ret)
	{
		ret = new vector<uint32_t>();
		return 0.0;
	}
	virtual pair<double, double> edgeDecode(AdjacentMatrix*&, vector<double>&) { return make_pair(0, 0); };
	void compare_edge_topK(AdjacentMatrix*& A, uint32_t K, double& are, double& aae)
	{
		are = aae = 0;
		auto ground_truth = A->get_edge_topk(K);
		for (auto& e : ground_truth)
		{
			auto predict = edge_query(e.x, e.y);
			are += abs(1.0 * predict - e.w) / e.w;
			aae += abs(1.0 * predict - e.w);
		}
		are /= K;
		aae /= K;
	}
	void compare_node_topK(AdjacentMatrix*& A, uint32_t K, double& are, double& aae)
	{
		are = aae = 0;
		auto ground_truth = A->get_node_topk(K);
		for (auto& pr : ground_truth)
		{
			auto predict = node_query(pr.second);
			are += abs(1.0 * predict - pr.first) / pr.first;
			aae += abs(1.0 * predict - pr.first);
		}
		are /= K;
		aae /= K;
	}
	uint64_t insert_time, edge_query_time, node_query_time, decode_time;
protected:
#ifdef WIN32

	void start_timing()
	{
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&t1);
	}
	double stop_timing()
	{
		QueryPerformanceCounter(&t2);
		return (double)(t2.QuadPart - t1.QuadPart) / (double)nFreq.QuadPart * 1000000000;
	}
	void start_timing(LARGE_INTEGER& t1)
	{
		QueryPerformanceCounter(&t1);
	}
	double stop_timing(LARGE_INTEGER t1)
	{
		LARGE_INTEGER t2, nFreq;
		QueryPerformanceCounter(&t2);
		QueryPerformanceFrequency(&nFreq);
		return (double)(t2.QuadPart - t1.QuadPart) / (double)nFreq.QuadPart * 1000000000;
	}
	LARGE_INTEGER nFreq, t1, t2;
#else
	void start_timing()
	{
		clock_gettime(CLOCK_MONOTONIC, &t1);
	}
	double stop_timing()
	{
		clock_gettime(CLOCK_MONOTONIC, &t2);
		return (t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec);
	}
	void start_timing(timespec& t1)
	{
		clock_gettime(CLOCK_MONOTONIC, &t1);
	}
	double stop_timing(timespec t1)
	{
		timespec t2;
		clock_gettime(CLOCK_MONOTONIC, &t2);
		return (t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec);
	}
	timespec t1, t2;
#endif

private:
	string name;
};
