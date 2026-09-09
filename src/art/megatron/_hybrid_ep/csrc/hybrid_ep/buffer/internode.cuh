// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once

#include <pybind11/pybind11.h>
#include "coordinator.cuh"
#include "config.cuh"
#include "backend/topo_detection.cuh"
#include "backend/hybrid_ep_backend.cuh"

#ifdef USE_NIXL
#include "buffer/internode_nixl.cuh"
#else
#include "buffer/internode_doca.cuh"
#endif
