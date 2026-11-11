#pragma once
class BloomFilterBase
{
public:
	BloomFilterBase() { }

	virtual ~BloomFilterBase() {}
	virtual inline void insert(uint64_t x, uint64_t y) = 0;
	virtual inline uint8_t query(uint64_t x, uint64_t y)  const = 0;
private:
};
