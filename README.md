# SBD-iPALM: Sparse blind deconvolution using iPALM

**SBD-iPALM** is a MATLAB package for *sparse blind deconvolution* (SBD) using the iPALM method, motivated by studies in blind deconvolution as a nonconvex optimization problem, and by applications in Scanning Tunneling Microscopy (STM). The iPALM method is an accelerated first-order method with optimal convergence guarantees to a stationary point when solving nonconvex optimization problems -- see [reference](https://arxiv.org/abs/1702.02505).

## Updates for v2

**2018-03-08**:

- `mkcdl.m` updated to work with multiple samples. Centering needs to be updated and commenting still needed.


**2018-03-05**:

- Generic iterators for solving various bilinear problems on the sphere -- e.g. `ipalm` -- now power problem instances of interest -- e.g. SBD, CDL.

- SBD and CDL now implemented as wrapper functions for constructing specific iterator instances.

- The `huber` loss is now rewritten to acommodate for the iterator structure. Reweighting is included.

- All iterating, reweighting and display functions that go on top of the inner loop can now be written seperately, e.g. this means one can run iterations until desired and not lose information. Changes to each problem setting can be made easily.


# Expected updates

- An ADMM iterator is expected soon.

- Reimplement wrappers of SBD, CDL, etc. as classes? To add various associated methods such as synthesis.