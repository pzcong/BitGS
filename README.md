# BitGS core implementation

Core graph sketch implementation with layered counter updates, recursive edge and node queries, Bloom filters, and graph reconstruction. The public sketch class and output name are `BitGS`; its configuration type is `BitGSLayerConfig`.

## Files

- `include/BitGS.h`: layered sketch and reconstruction.
- `include/SketchBase.h`: query interface and timing helpers.
- `include/BloomFilter/`: membership filters and overflow bitmap.
- `include/common/`: hashing, packed counters, and graph data structures.
- `examples/basic.cpp`: synthetic insertion/query example.


## Build and run

```sh
make EIGEN_INCLUDE=/absolute/path/to/eigen3
make run
```
