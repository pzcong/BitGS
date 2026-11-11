#pragma once
#include "BloomFilterBase.h"
class BloomFilter :public BloomFilterBase
{
public:
	BloomFilter(uint32_t memory_in_byte, uint32_t _hash_num)
	{
		LENGTH = memory_in_byte * 8;
		array = new uint32_t[memory_in_byte / sizeof(uint32_t)];
		memset(array, 0, memory_in_byte);
		hash_num = _hash_num;
	}
	virtual ~BloomFilter()
	{
		delete[] array;
	}
	void insert(uint64_t x, uint64_t y)
	{
		uint64_t q = (x << 32) + y;
		for (uint32_t i = 0; i < hash_num; i++)
		{
			uint32_t p = hash32(q, i) % LENGTH;
			array[p >> 5] |= 1 << (p & 31);
		}
	}
	uint8_t query(uint64_t x, uint64_t y) const
	{
		uint8_t ret = 1;
		uint64_t q = (x << 32) + y;
		for (uint32_t i = 0; i < hash_num && ret; i++)
		{
			uint32_t p = hash32(q, i) % LENGTH;
			ret &= array[p >> 5] >> (p & 31);
		}
		return ret;
	}
	uint32_t* array;
	uint32_t hash_num, LENGTH, xLENGTH, yLENGTH, DIMENSION;
};
