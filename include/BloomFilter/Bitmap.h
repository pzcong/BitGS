#pragma once
#include "BloomFilterBase.h"
#include<unordered_set>
class Bitmap :public BloomFilterBase
{
public:
	Bitmap(uint64_t _width) :width(_width)
	{
		uint64_t len = (width * width - 1) / 32 + 1;
		mp = new uint32_t[len];
		memset(mp, 0, len * 4);
	}
	virtual ~Bitmap()
	{
		delete[] mp;
	}
	virtual void insert(uint64_t x, uint64_t y)
	{
		uint64_t offset = x * width + y;
		mp[offset >> 5] |= 1 << (offset & 31);
	}
	virtual uint8_t query(uint64_t x, uint64_t y) const
	{
		uint64_t offset = x * width + y;
		return (mp[offset >> 5] >> (offset & 31)) & 1;
	}
	uint32_t* mp;
	uint64_t width;
};
