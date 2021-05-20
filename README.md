# o1heap

O1heap is a highly deterministic constant-complexity memory allocator designed for hard real-time high-integrity
embedded systems. The name stands for O(1) heap.

This Rust implementation is closely based on [Pavel Kirienko's original C implementation](https://github.com/pavel-kirienko/o1heap).
I only translated it into Rust. Please see the readme there for details about how this allocator works.

## License

MIT license (see LICENSE.txt)

The original C implementation is also MIT-licensed:

```
Copyright (c) 2020 Pavel Kirienko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

