// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#ifdef USE_NIXL

#include "buffer/nixl_connector.h"
#include "backend/hybrid_ep_backend.cuh"
#include <fstream>
#include <thread>
#include <chrono>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <stdexcept>

#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000)  // 1000 seconds

// Uncomment to enable detailed debug logging.
// #define NIXL_VERBOSE

#ifdef NIXL_VERBOSE
  #define NIXL_LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
  #define NIXL_LOG(fmt, ...) do {} while(0)
#endif

#define NIXL_LOG_CRITICAL(fmt, ...) printf(fmt, ##__VA_ARGS__)

namespace hybrid_ep {

static void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// getenv("NAME") parsed as int; returns default_val if unset or invalid.
static int getenv_int(const char* name, int default_val) {
    const char* s = std::getenv(name);
    if (!s || !*s) {
        return default_val;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s) {
        return default_val;
    }
    return static_cast<int>(v);
}

static std::string get_nixl_run_id() {
    for (const char* var : {"DEEPEP_NIXL_RUN_ID", "SLURM_STEP_ID", "SLURM_JOB_ID"}) {
        const char* v = std::getenv(var);
        if (v && *v) return std::string(v);
    }
    return "";
}

// makeConnection can return NOT_FOUND until the comm thread finishes applying remote UCX
// connection info; at multi-node scale that may lag slightly after checkRemoteMD succeeds.
static bool nixl_wireup_status_retriable(nixl_status_t s) {
    return s == NIXL_ERR_NOT_FOUND || s == NIXL_ERR_BACKEND;
}

static bool prep_mem_view_status_retriable(nixl_status_t s) {
    return s == NIXL_ERR_NOT_FOUND || s == NIXL_ERR_BACKEND;
}

// Local prepMemView can briefly fail while local VRAM registration is still settling.
static nixl_status_t prep_local_mem_view_retry(std::shared_ptr<nixlAgent> agent,
                                               const nixl_xfer_dlist_t& dlist,
                                               nixlMemViewH& mvh,
                                               nixl_opt_args_t* extra_params,
                                               int rank_uuid,
                                               const char* what) {
    const int max_retries = std::max(1, getenv_int("DEEPEP_NIXL_PREPMV_MAX_RETRIES", 5000));
    const int retry_ms = std::max(1, getenv_int("DEEPEP_NIXL_PREPMV_RETRY_MS", 20));
    nixl_status_t status = NIXL_ERR_UNKNOWN;
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        status = agent->prepMemView(dlist, mvh, extra_params);
        if (status == NIXL_SUCCESS) {
            return NIXL_SUCCESS;
        }
        if (!prep_mem_view_status_retriable(status)) {
            break;
        }
        if (attempt == 0 || (attempt + 1) % 100 == 0) {
            NIXL_LOG_CRITICAL(
                "  [Rank %d] prepMemView(local) %s: pending (%s), attempt %d/%d\n",
                rank_uuid,
                what,
                nixlEnumStrings::statusStr(status).c_str(),
                attempt + 1,
                max_retries);
        }
        sleep_ms(retry_ms);
    }
    return status;
}

// Remote prepMemView can fail with NOT_FOUND until remoteSections + UCX rkeys fully match
// the descriptors (comm thread / etcd lag at scale). Retry similarly to makeConnection.
static nixl_status_t prep_remote_mem_view_retry(std::shared_ptr<nixlAgent> agent,
                                                const nixl_remote_dlist_t& dlist,
                                                nixlMemViewH& mvh,
                                                nixl_opt_args_t* extra_params,
                                                int rank_uuid,
                                                const char* what) {
    const int max_retries = std::max(1, getenv_int("DEEPEP_NIXL_PREPMV_MAX_RETRIES", 5000));
    const int retry_ms = std::max(1, getenv_int("DEEPEP_NIXL_PREPMV_RETRY_MS", 20));
    nixl_status_t status = NIXL_ERR_UNKNOWN;
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        status = agent->prepMemView(dlist, mvh, extra_params);
        if (status == NIXL_SUCCESS) {
            return NIXL_SUCCESS;
        }
        if (!prep_mem_view_status_retriable(status)) {
            break;
        }
        if (attempt == 0 || (attempt + 1) % 100 == 0) {
            NIXL_LOG_CRITICAL(
                "  [Rank %d] prepMemView(remote) %s: pending (%s), attempt %d/%d\n",
                rank_uuid,
                what,
                nixlEnumStrings::statusStr(status).c_str(),
                attempt + 1,
                max_retries);
        }
        sleep_ms(retry_ms);
    }
    return status;
}

static std::string get_local_ip() {
    struct ifaddrs *ifaddr, *ifa;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return "127.0.0.1";
    }

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET)
            continue;

        if ((ifa->ifa_flags & IFF_UP) && !(ifa->ifa_flags & IFF_LOOPBACK)) {
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST) == 0) {
                freeifaddrs(ifaddr);
                return std::string(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return "127.0.0.1";
}

static std::string boot_id_get() {
    std::ifstream boot_id_file("/proc/sys/kernel/random/boot_id");
    if (!boot_id_file.is_open()) {
        return "";
    }

    std::string boot_id;
    std::getline(boot_id_file, boot_id);

    if (!boot_id.empty() && boot_id.back() == '\n') {
        boot_id.pop_back();
    }

    return boot_id;
}

static ino_t ipc_namespace_inode_get() {
    struct stat st;
    if (stat("/proc/self/ns/ipc", &st) != 0) {
        return 0;
    }
    return st.st_ino;
}

HybridEP_NIXLConnector::HybridEP_NIXLConnector(int rank_uuid, int local_device_id)
    : rank_uuid(rank_uuid),
      local_device_id(local_device_id),
      num_ranks(0),
      num_experts_per_rank(0),
      num_nodes(0),
      ranks_per_node(0),
      num_channels(0),
      initialized(false),
      connected(false),
      d_dispatch_nixl_ctx(nullptr),
      d_combine_nixl_ctx(nullptr),
      current_epoch(0),
      gda_num_channels(0),
      d_dispatch_flag_counters(nullptr),
      d_combine_flag_counters(nullptr) {

    NIXL_LOG("  [Rank %d] HybridEP_NIXLConnector: Constructor called (local_device_id=%d)\n", rank_uuid, local_device_id);

    my_peer_info = {};
    strncpy(my_peer_info.ip, get_local_ip().c_str(), MAX_IP_LENGTH - 1);
    my_peer_info.ip[MAX_IP_LENGTH - 1] = '\0';
    strncpy(my_peer_info.boot_id, boot_id_get().c_str(), MAX_BOOT_ID_LENGTH - 1);
    my_peer_info.boot_id[MAX_BOOT_ID_LENGTH - 1] = '\0';
    my_peer_info.ipc_namespace_inode = ipc_namespace_inode_get();
    my_peer_info.device_id = local_device_id;
    my_peer_info.rank = rank_uuid;

    NIXL_LOG("  [Rank %d] HybridEP_NIXLConnector: My info - IP=%s, device_id=%d, boot_id=%s\n",
           rank_uuid, my_peer_info.ip, my_peer_info.device_id, my_peer_info.boot_id);
}

HybridEP_NIXLConnector::~HybridEP_NIXLConnector() {
    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Destructor called - cleaning up resources...\n", rank_uuid);

    cudaDeviceSynchronize();

    if (connected && !connected_ranks.empty()) {
        NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Disconnecting %zu ranks...\n", rank_uuid, connected_ranks.size());
        disconnectRanks(connected_ranks);
        connected = false;
    }

    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Destroying NIXL agents...\n", rank_uuid);
    for (auto& agent_info : nixl_agent_infos) {
        agent_info.agent.reset();
    }
    nixl_agent_infos.clear();
    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: NIXL agents destroyed\n", rank_uuid);

    cudaDeviceSynchronize();

    if (d_dispatch_flag_counters) {
        cudaFree(d_dispatch_flag_counters);
        d_dispatch_flag_counters = nullptr;
    }
    if (d_combine_flag_counters) {
        cudaFree(d_combine_flag_counters);
        d_combine_flag_counters = nullptr;
    }

    if (d_dispatch_nixl_ctx) {
        cudaFree(d_dispatch_nixl_ctx);
        d_dispatch_nixl_ctx = nullptr;
    }
    if (d_combine_nixl_ctx) {
        cudaFree(d_combine_nixl_ctx);
        d_combine_nixl_ctx = nullptr;
    }

    cudaDeviceSynchronize();

    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Cleanup complete\n", rank_uuid);
}

void HybridEP_NIXLConnector::updateMemoryBuffers(
    int num_ranks, int num_experts_per_rank, int num_nodes, int ranks_per_node,
    int num_dispatch_blocks, int num_combine_blocks,
    InterNodeDispatchBuffers& dispatch_buffers,
    InterNodeCombineBuffers& combine_buffers) {

    NIXL_LOG("  [Rank %d] updateMemoryBuffers: Validating parameters...\n", rank_uuid);
    assert(rank_uuid >= 0 && rank_uuid < num_ranks && "Invalid rank_uuid");
    assert(!initialized && "updateMemoryBuffers can only be called once");
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: Parameters valid\n", rank_uuid);

    this->num_ranks = num_ranks;
    this->num_experts_per_rank = num_experts_per_rank;
    this->num_nodes = num_nodes;
    this->ranks_per_node = ranks_per_node;

    this->num_channels = std::max(num_dispatch_blocks, num_combine_blocks);
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: num_channels=%d\n", rank_uuid, num_channels);

    dispatch_buf = &dispatch_buffers;
    combine_buf = &combine_buffers;
    forward_dispatch = true;
    backward_combine = true;
    use_fp8 = (dispatch_buffers.data_type == APP_TOKEN_DATA_TYPE::UINT8);

    my_peer_info.rdma_buffer_ptr = dispatch_buffers.rdma_inter_node_group_token;
    my_peer_info.rdma_prob_buffer_ptr = dispatch_buffers.rdma_inter_node_group_prob;
    my_peer_info.rdma_scaling_factor_buffer_ptr = dispatch_buffers.rdma_inter_node_group_scaling_factor;
    my_peer_info.dispatch_flags_ptr = dispatch_buffers.rdma_inter_node_group_flags;
    my_peer_info.combine_rdma_buffer_ptr = combine_buffers.rdma_inter_node_group_token;
    my_peer_info.combine_rdma_prob_buffer_ptr = combine_buffers.rdma_inter_node_group_prob;
    my_peer_info.combine_flags_ptr = combine_buffers.rdma_inter_node_group_flags;
    my_peer_info.device_id = local_device_id;
    my_peer_info.rank = rank_uuid;
    nixl_peer_info.resize(num_ranks);

    initialized = true;
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: All buffer pointers stored\n", rank_uuid);

    NIXL_LOG("  [Rank %d] updateMemoryBuffers: Creating NIXL agent and registering buffers...\n", rank_uuid);
    _nixl_agents_init(1);
    _register_buffers_with_agents();
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: NIXL agent created and buffers registered\n", rank_uuid);
}

void HybridEP_NIXLConnector::connectRanks(const std::vector<int>& remote_rank_uuids) {
    NIXL_LOG("  [Rank %d] connectRanks: Starting connection process...\n", rank_uuid);
    assert(initialized && "Must call updateMemoryBuffers before connectRanks");
    assert(!connected && "connectRanks can only be called once");
    assert(num_channels > 0 && "num_channels must be set in updateMemoryBuffers");

    std::vector<int> ranks_to_connect;
    for (int remote_rank : remote_rank_uuids) {
        if (remote_rank != rank_uuid) {
            ranks_to_connect.push_back(remote_rank);
        }
    }

    if (ranks_to_connect.empty()) {
        NIXL_LOG("  [Rank %d] connectRanks: No remote ranks to connect to\n", rank_uuid);
        connected = true;
        return;
    }

    NIXL_LOG("  [Rank %d] connectRanks: Connecting to %zu remote agents\n", rank_uuid, ranks_to_connect.size());
    _nixl_agents_connect(ranks_to_connect);

    NIXL_LOG("  [Rank %d] connectRanks: Peer info exchange\n", rank_uuid);
    _nixl_agents_wireup(ranks_to_connect);

    NIXL_LOG("  [Rank %d] connectRanks: Creating memory views\n", rank_uuid);
    _nixl_create_memory_views(ranks_to_connect);

    NIXL_LOG("  [Rank %d] connectRanks: Building GPU contexts\n", rank_uuid);
    _nixl_build_gpu_contexts(num_channels, num_channels);

    connected_ranks = ranks_to_connect;
    connected = true;

    NIXL_LOG("  [Rank %d] connectRanks: Successfully connected to %zu ranks\n", rank_uuid, connected_ranks.size());
}

void HybridEP_NIXLConnector::disconnectRanks(const std::vector<int>& remote_rank_uuids) {
    assert(connected && "Must be connected before disconnecting");
    _nixl_agents_wiredown(remote_rank_uuids);
}

dispatch_gpu_nixl_ctx* HybridEP_NIXLConnector::get_dispatch_gpu_ctx() {
    return d_dispatch_nixl_ctx;
}

combine_gpu_nixl_ctx* HybridEP_NIXLConnector::get_combine_gpu_ctx() {
    return d_combine_nixl_ctx;
}

void HybridEP_NIXLConnector::_nixl_agents_init(int num_agents) {
    NIXL_LOG("    [Rank %d] _nixl_agents_init: creating %d agent(s)\n", rank_uuid, num_agents);
    nixl_agent_infos.clear();
    nixl_agent_infos.reserve(num_agents);

    static int nixl_epoch = 0;
    current_epoch = nixl_epoch++;
    nixl_run_id = get_nixl_run_id();
    std::string agent_name = std::to_string(rank_uuid) + "_e" + std::to_string(current_epoch);
    if (!nixl_run_id.empty()) agent_name += "_r" + nixl_run_id;

    const char* etcd_endpoint = std::getenv("NIXL_ETCD_ENDPOINTS");
    if (etcd_endpoint) {
        NIXL_LOG("    [Rank %d] _nixl_agents_init: etcd=%s\n", rank_uuid, etcd_endpoint);
    } else {
        NIXL_LOG("    [Rank %d] _nixl_agents_init: NIXL_ETCD_ENDPOINTS not set, using default\n", rank_uuid);
    }

    nixlAgentConfig cfg(true, false, 0,
                       nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 1, 0, 100000, false, NIXL_ETCD_WATCH_TIMEOUT);

    auto agent = std::make_shared<nixlAgent>(agent_name, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    nixl_status_t status = agent->getPluginParams("UCX", mems, init_params);
    assert(status == NIXL_SUCCESS);

    init_params["ucx_error_handling_mode"] = "none";
    init_params["num_workers"] = std::to_string(1);

    gda_num_channels = std::max(1, getenv_int("DEEPEP_NIXL_GDA_NUM_CHANNELS", num_channels));
    init_params["ucx_num_device_channels"] = std::to_string(gda_num_channels);
    NIXL_LOG("    [Rank %d] _nixl_agents_init: gda_num_channels=%d\n", rank_uuid, gda_num_channels);

    nixlBackendH* ucx_backend = nullptr;
    status = agent->createBackend("UCX", init_params, ucx_backend);
    assert(status == NIXL_SUCCESS && ucx_backend != nullptr);

    int num_remote_nodes = num_nodes - 1;

    nixl_agent_infos.emplace_back(num_remote_nodes, num_ranks);
    nixl_agent_infos[0].agent = agent;
    nixl_agent_infos[0].agent_name = agent_name;
    nixl_agent_infos[0].backend = ucx_backend;
    nixl_agent_infos[0].extra_params.backends.push_back(ucx_backend);
    NIXL_LOG("    [Rank %d] _nixl_agents_init: agent=%s (%d remote nodes)\n",
             rank_uuid, agent_name.c_str(), num_remote_nodes);
}

void HybridEP_NIXLConnector::_nixl_agents_connect(const std::vector<int>& ranks) {
    NIXL_LOG("    [Rank %d] _nixl_agents_connect: Connecting to %zu remote agents...\n", rank_uuid, ranks.size());
    int agent_idx = 0;

    // sendLocalMD is async — metadata may not be visible in etcd immediately after
    // the caller's barrier.  Give the etcd watch time to fire before invalidating;
    // aggressive invalidation cancels the active watch and can miss the arrival event.
    const int refetch_interval = std::max(1, getenv_int("DEEPEP_NIXL_FETCH_RETRY_INTERVAL", 3000));
    const int max_fetch_retries = std::max(1, getenv_int("DEEPEP_NIXL_FETCH_MAX_RETRIES", 10));
    const int hard_timeout_ms   = std::max(1000, getenv_int("DEEPEP_NIXL_CONNECT_TIMEOUT_MS", 120000));

    for (int remote_rank : ranks) {
        std::string remote_agent_name = std::to_string(remote_rank) + "_e" + std::to_string(current_epoch);
        if (!nixl_run_id.empty()) remote_agent_name += "_r" + nixl_run_id;
        nixl_agent_infos[agent_idx].dst_agent_names[remote_rank] = remote_agent_name;

        auto& agent = nixl_agent_infos[agent_idx].agent;
        auto& extra = nixl_agent_infos[agent_idx].extra_params;

        NIXL_LOG("    [Rank %d] _nixl_agents_connect: Fetching metadata for remote agent '%s'...\n",
               rank_uuid, remote_agent_name.c_str());
        nixl_status_t fetch_status = agent->fetchRemoteMD(remote_agent_name, &extra);
        assert(fetch_status == NIXL_SUCCESS);

        nixl_xfer_dlist_t empty_descs(VRAM_SEG);
        int wait_count = 0;
        int fetch_retries = 0;
        auto t_start = std::chrono::steady_clock::now();
        while (agent->checkRemoteMD(remote_agent_name, empty_descs) != NIXL_SUCCESS) {
            sleep_ms(10);
            wait_count++;

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            if (elapsed > hard_timeout_ms) {
                fprintf(stderr,
                    "ERROR: [Rank %d] _nixl_agents_connect: timed out after %ld ms waiting for "
                    "remote agent '%s' (epoch %d). Remote rank may not have published metadata.\n",
                    rank_uuid, (long)elapsed, remote_agent_name.c_str(), current_epoch);
                fflush(stderr);
                assert(false && "NIXL metadata fetch timed out");
            }

            if (wait_count % 500 == 0) {
                NIXL_LOG("    [Rank %d] _nixl_agents_connect: Still waiting for rank %d metadata (%d waits, %ld ms)...\n",
                       rank_uuid, remote_rank, wait_count, (long)elapsed);
            }
            if (wait_count > 0 && wait_count % refetch_interval == 0 && fetch_retries < max_fetch_retries) {
                fetch_retries++;
                NIXL_LOG_CRITICAL(
                    "  [Rank %d] _nixl_agents_connect: checkRemoteMD for %s still failing after "
                    "%d waits, invalidating stale state and re-fetching (retry %d/%d)\n",
                    rank_uuid, remote_agent_name.c_str(), wait_count, fetch_retries, max_fetch_retries);
                agent->invalidateRemoteMD(remote_agent_name);
                fetch_status = agent->fetchRemoteMD(remote_agent_name, &extra);
                assert(fetch_status == NIXL_SUCCESS);
            }
        }
        NIXL_LOG("    [Rank %d] _nixl_agents_connect: Metadata available from rank %d\n", rank_uuid, remote_rank);
    }
    NIXL_LOG("    [Rank %d] _nixl_agents_connect: Remote metadata fetched for all %zu ranks\n", rank_uuid, ranks.size());
}

void HybridEP_NIXLConnector::_nixl_agents_wireup(const std::vector<int>& ranks) {
    NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Starting wireup for %zu ranks...\n", rank_uuid, ranks.size());
    int agent_idx = 0;

    for (int remote_rank : ranks) {
        const std::string& remote_agent_name = nixl_agent_infos[agent_idx].dst_agent_names[remote_rank];
        std::string my_peer_info_str(reinterpret_cast<const char*>(&my_peer_info), sizeof(NixlPeerInfo));
        nixl_agent_infos[agent_idx].agent->genNotif(remote_agent_name, my_peer_info_str);
        NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Sent peer info notification to rank %d\n", rank_uuid, remote_rank);
    }

    int received_count = 0;
    for (int remote_rank : ranks) {
        int poll_count = 0;
        while (!nixl_agent_infos[agent_idx].wire_up_done[remote_rank]) {
            nixl_notifs_t notif_map;
            nixl_agent_infos[agent_idx].agent->getNotifs(notif_map);

            for (auto &notif : notif_map) {
                std::string peer_info_payload = notif.second[0];
                NixlPeerInfo remote_peer_info;
                memcpy(&remote_peer_info, peer_info_payload.c_str(), sizeof(NixlPeerInfo));
                nixl_peer_info[remote_peer_info.rank] = remote_peer_info;
                nixl_agent_infos[agent_idx].wire_up_done[remote_peer_info.rank] = true;
                received_count++;

                NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Received peer info from rank %d (%d/%zu)\n",
                       rank_uuid, remote_peer_info.rank, received_count, ranks.size());
            }

            poll_count++;
            if (poll_count % 1000 == 0) {
                NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Still waiting for rank %d (%d polls)...\n",
                       rank_uuid, remote_rank, poll_count);
            }
            sleep_ms(1);
        }
    }
    _nixl_ucx_wireup(ranks);
    NIXL_LOG("    [Rank %d] _nixl_agents_wireup: done\n", rank_uuid);
}

void HybridEP_NIXLConnector::_nixl_ucx_wireup(const std::vector<int>& ranks) {
    int agent_idx = 0;
    const int max_retries = std::max(1, getenv_int("DEEPEP_NIXL_WIREUP_MAX_RETRIES", 2000));
    const int retry_ms = std::max(1, getenv_int("DEEPEP_NIXL_WIREUP_RETRY_MS", 10));

    for (int remote_rank : ranks) {
        const std::string& remote_agent_name = nixl_agent_infos[agent_idx].dst_agent_names[remote_rank];

        nixl_status_t status = NIXL_ERR_UNKNOWN;
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            status = nixl_agent_infos[agent_idx].agent->makeConnection(
                remote_agent_name, &nixl_agent_infos[agent_idx].extra_params);
            if (status == NIXL_SUCCESS) {
                break;
            }
            if (!nixl_wireup_status_retriable(status)) {
                break;
            }
            if (attempt == 0 || (attempt + 1) % 100 == 0) {
                NIXL_LOG_CRITICAL(
                    "  [Rank %d] _nixl_ucx_wireup: makeConnection to agent %s pending (%s), "
                    "attempt %d/%d\n",
                    rank_uuid,
                    remote_agent_name.c_str(),
                    nixlEnumStrings::statusStr(status).c_str(),
                    attempt + 1,
                    max_retries);
            }
            sleep_ms(retry_ms);
        }

        if (status != NIXL_SUCCESS) {
            std::string msg = std::string("HybridEP NIXL: makeConnection failed for remote agent ") +
                remote_agent_name + ": " + nixlEnumStrings::statusStr(status) +
                " (code " + std::to_string(static_cast<int>(status)) + "). "
                "Try increasing DEEPEP_NIXL_WIREUP_MAX_RETRIES or DEEPEP_NIXL_WIREUP_RETRY_MS "
                "if etcd/UCX is slow at scale.";
            NIXL_LOG_CRITICAL("%s\n", msg.c_str());
            throw std::runtime_error(msg);
        }

        NIXL_LOG("    [Rank %d] _nixl_ucx_wireup: Connected to rank %d\n", rank_uuid, remote_rank);
    }
}

void HybridEP_NIXLConnector::_nixl_agents_wiredown(const std::vector<int>& ranks) {
    int agent_idx = 0;

    for (int remote_rank : ranks) {
        nixl_status_t status = nixl_agent_infos[agent_idx].agent->invalidateRemoteMD(
            nixl_agent_infos[agent_idx].dst_agent_names[remote_rank]);
        if (status != NIXL_SUCCESS && status != NIXL_ERR_NOT_FOUND) {
            fprintf(stderr, "WARNING: Failed to invalidate remote metadata for rank %d\n", remote_rank);
        }
        nixl_agent_infos[agent_idx].dst_agent_names[remote_rank].clear();
        nixl_agent_infos[agent_idx].wire_up_done[remote_rank] = false;
    }
}

void HybridEP_NIXLConnector::_nixl_create_memory_views(const std::vector<int>& ranks) {
    NIXL_LOG("    [Rank %d] _nixl_create_memory_views: Creating memory views...\n", rank_uuid);

    int agent_idx = 0;
    int node_rank = rank_uuid / ranks_per_node;
    int local_nvl_rank = rank_uuid % ranks_per_node;
    int num_remote_nodes = num_nodes - 1;

    for (int peer_idx = 0; peer_idx < num_remote_nodes; ++peer_idx) {
        int actual_node_rank = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
        int expected_remote_rank = actual_node_rank * ranks_per_node + local_nvl_rank;
        bool found = std::find(ranks.begin(), ranks.end(), expected_remote_rank) != ranks.end();
        assert(found && "Expected remote rank not found in connected ranks list");
    }

    size_t token_stride = dispatch_buf->attn_input_token_sz;
    NIXL_LOG("    [Rank %d]   Token stride=%zu bytes\n", rank_uuid, token_stride);

    // -- Dispatch memory views --
    NIXL_LOG("    [Rank %d]   Creating dispatch memory views...\n", rank_uuid);
    nixl_xfer_dlist_t dispatch_local_descs(VRAM_SEG);
    if (dispatch_buf->attn_input_token && dispatch_buf->attn_input_token_sz > 0) {
        dispatch_local_descs.addDesc(nixlBlobDesc((uintptr_t)dispatch_buf->attn_input_token,
                                                   dispatch_buf->attn_input_token_sz, local_device_id, ""));
    }

    if (dispatch_buf->attn_input_prob && forward_dispatch) {
        dispatch_local_descs.addDesc(nixlBlobDesc((uintptr_t)dispatch_buf->attn_input_prob,
                                                   dispatch_buf->attn_input_prob_sz, local_device_id, ""));
    }

    if (dispatch_buf->attn_input_scaling_factor && use_fp8) {
        dispatch_local_descs.addDesc(nixlBlobDesc((uintptr_t)dispatch_buf->attn_input_scaling_factor,
                                                   dispatch_buf->attn_input_scaling_factor_sz, local_device_id, ""));
    }

    nixl_remote_dlist_t dispatch_remote_data_descs(VRAM_SEG);
    nixl_remote_dlist_t dispatch_remote_signal_descs(VRAM_SEG);

    for (int peer_idx = 0; peer_idx < num_remote_nodes; ++peer_idx) {
        int actual_node_rank = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
        int remote_rank = actual_node_rank * ranks_per_node + local_nvl_rank;

        int my_node_rank_in_remote = (node_rank < actual_node_rank) ? node_rank : (node_rank - 1);
        const std::string& remote_agent_name = nixl_agent_infos[agent_idx].dst_agent_names[remote_rank];

        void* remote_dispatch_token_addr = (uint8_t*)nixl_peer_info[remote_rank].rdma_buffer_ptr +
                                           my_node_rank_in_remote * token_stride;
        dispatch_remote_data_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_dispatch_token_addr,
            token_stride,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        if (dispatch_buf->rdma_inter_node_group_prob && forward_dispatch) {
            size_t prob_stride = dispatch_buf->rdma_inter_node_group_prob_sz/(num_nodes-1);
            void* remote_dispatch_prob_addr = (uint8_t*)nixl_peer_info[remote_rank].rdma_prob_buffer_ptr +
                                                my_node_rank_in_remote * prob_stride;
            dispatch_remote_data_descs.addDesc(nixlRemoteDesc(
                (uintptr_t)remote_dispatch_prob_addr,
                prob_stride,
                nixl_peer_info[remote_rank].device_id,
                remote_agent_name));
        }

        if (dispatch_buf->rdma_inter_node_group_scaling_factor && use_fp8) {
            size_t sf_stride = dispatch_buf->attn_input_scaling_factor_sz;
            void* remote_dispatch_scaling_factor_addr = (uint8_t*)nixl_peer_info[remote_rank].rdma_scaling_factor_buffer_ptr +
                                                my_node_rank_in_remote * sf_stride;
            dispatch_remote_data_descs.addDesc(nixlRemoteDesc(
                (uintptr_t)remote_dispatch_scaling_factor_addr,
                sf_stride,
                nixl_peer_info[remote_rank].device_id,
                remote_agent_name));
        }

        uint64_t* remote_dispatch_flag_addr = nixl_peer_info[remote_rank].dispatch_flags_ptr;
        dispatch_remote_signal_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_dispatch_flag_addr,
            dispatch_buf->rdma_inter_node_group_flags_sz,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        NIXL_LOG("    [Rank %d]     dispatch[%d] -> remote_rank=%d, data=%p, signal=%p\n",
               rank_uuid, peer_idx, remote_rank, remote_dispatch_token_addr, (void*)remote_dispatch_flag_addr);
    }

    nixl_status_t status;
    status = prep_local_mem_view_retry(
        nixl_agent_infos[agent_idx].agent,
        dispatch_local_descs,
        nixl_agent_infos[agent_idx].dispatch_local_mvh,
        &nixl_agent_infos[agent_idx].extra_params,
        rank_uuid,
        "dispatch_local");
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: prepMemView dispatch local failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    {
        const int pre_delay = getenv_int("DEEPEP_NIXL_PREPMV_INITIAL_DELAY_MS", 0);
        if (pre_delay > 0) {
            NIXL_LOG_CRITICAL(
                "  [Rank %d] DEEPEP_NIXL_PREPMV_INITIAL_DELAY_MS: sleeping %d ms before remote views\n",
                rank_uuid,
                pre_delay);
            sleep_ms(pre_delay);
        }
    }

    status = prep_remote_mem_view_retry(
        nixl_agent_infos[agent_idx].agent,
        dispatch_remote_data_descs,
        nixl_agent_infos[agent_idx].dispatch_remote_data_mvh,
        &nixl_agent_infos[agent_idx].extra_params,
        rank_uuid,
        "dispatch_remote_data");
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: prepMemView dispatch remote data failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) +
            "). Increase DEEPEP_NIXL_PREPMV_MAX_RETRIES / DEEPEP_NIXL_PREPMV_RETRY_MS if etcd is slow.");
    }

    status = prep_remote_mem_view_retry(
        nixl_agent_infos[agent_idx].agent,
        dispatch_remote_signal_descs,
        nixl_agent_infos[agent_idx].dispatch_remote_signal_mvh,
        &nixl_agent_infos[agent_idx].extra_params,
        rank_uuid,
        "dispatch_remote_signal");
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: prepMemView dispatch remote signal failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    NIXL_LOG("    [Rank %d]   Dispatch memory views created\n", rank_uuid);

    // -- Combine memory views --
    NIXL_LOG("    [Rank %d]   Creating combine memory views...\n", rank_uuid);

    nixl_xfer_dlist_t combine_local_descs(VRAM_SEG);
    if (combine_buf->rdma_intra_node_red_token && combine_buf->rdma_intra_node_red_token_sz > 0) {
        combine_local_descs.addDesc(nixlBlobDesc((uintptr_t)combine_buf->rdma_intra_node_red_token,
                                                  combine_buf->rdma_intra_node_red_token_sz, local_device_id, ""));
    }
    if (combine_buf->rdma_intra_node_red_prob && backward_combine) {
        combine_local_descs.addDesc(nixlBlobDesc((uintptr_t)combine_buf->rdma_intra_node_red_prob,
                                                  combine_buf->rdma_intra_node_red_prob_sz, local_device_id, ""));
    }

    nixl_remote_dlist_t combine_remote_data_descs(VRAM_SEG);
    nixl_remote_dlist_t combine_remote_signal_descs(VRAM_SEG);

    for (int peer_idx = 0; peer_idx < num_remote_nodes; ++peer_idx) {
        int actual_node_rank = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
        int remote_rank = actual_node_rank * ranks_per_node + local_nvl_rank;
        int my_node_rank_in_remote = (node_rank < actual_node_rank) ? node_rank : (node_rank - 1);
        const std::string& remote_agent_name = nixl_agent_infos[agent_idx].dst_agent_names[remote_rank];

        size_t combine_token_stride = combine_buf->rdma_intra_node_red_token_sz / (num_nodes - 1);

        void* remote_combine_token_addr = (uint8_t*)nixl_peer_info[remote_rank].combine_rdma_buffer_ptr +
                                          my_node_rank_in_remote * combine_token_stride;
        combine_remote_data_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_combine_token_addr,
            combine_token_stride,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        if (combine_buf->rdma_inter_node_group_prob && backward_combine) {
            size_t combine_prob_stride = combine_buf->rdma_inter_node_group_prob_sz/(num_nodes-1);

            void* remote_combine_prob_addr = (uint8_t*)nixl_peer_info[remote_rank].combine_rdma_prob_buffer_ptr +
                                              my_node_rank_in_remote * combine_prob_stride;
            combine_remote_data_descs.addDesc(nixlRemoteDesc(
                (uintptr_t)remote_combine_prob_addr,
                combine_prob_stride,
                nixl_peer_info[remote_rank].device_id,
                remote_agent_name));

        }

        uint64_t* remote_combine_flag_addr = nixl_peer_info[remote_rank].combine_flags_ptr;
        combine_remote_signal_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_combine_flag_addr,
            combine_buf->rdma_inter_node_group_flags_sz,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        NIXL_LOG("    [Rank %d]     combine[%d] -> remote_rank=%d, data=%p, signal=%p\n",
               rank_uuid, peer_idx, remote_rank, remote_combine_token_addr, (void*)remote_combine_flag_addr);
    }

    status = prep_local_mem_view_retry(
        nixl_agent_infos[agent_idx].agent,
        combine_local_descs,
        nixl_agent_infos[agent_idx].combine_local_mvh,
        &nixl_agent_infos[agent_idx].extra_params,
        rank_uuid,
        "combine_local");
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: prepMemView combine local failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    status = prep_remote_mem_view_retry(
        nixl_agent_infos[agent_idx].agent,
        combine_remote_data_descs,
        nixl_agent_infos[agent_idx].combine_remote_data_mvh,
        &nixl_agent_infos[agent_idx].extra_params,
        rank_uuid,
        "combine_remote_data");
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: prepMemView combine remote data failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    status = prep_remote_mem_view_retry(
        nixl_agent_infos[agent_idx].agent,
        combine_remote_signal_descs,
        nixl_agent_infos[agent_idx].combine_remote_signal_mvh,
        &nixl_agent_infos[agent_idx].extra_params,
        rank_uuid,
        "combine_remote_signal");
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: prepMemView combine remote signal failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    NIXL_LOG("    [Rank %d]   Combine memory views created\n", rank_uuid);

    NIXL_LOG("    [Rank %d] _nixl_create_memory_views: All memory views created for %d remote nodes\n",
           rank_uuid, num_remote_nodes);
}

void HybridEP_NIXLConnector::_nixl_build_gpu_contexts(int num_dispatch_blocks, int num_combine_blocks) {
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: Building GPU contexts...\n", rank_uuid);

    int agent_idx = 0;
    int num_remote_nodes = num_nodes - 1;

    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: gda_num_channels=%d\n", rank_uuid, gda_num_channels);

    // -- Dispatch context --
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: Building dispatch GPU context...\n", rank_uuid);
    cudaMalloc(&d_dispatch_flag_counters, sizeof(uint64_t) * num_remote_nodes);
    cudaMemset(d_dispatch_flag_counters, 0, sizeof(uint64_t) * num_remote_nodes);

    int dispatch_local_stride = 1 + (int)forward_dispatch + (int)use_fp8;
    int dispatch_remote_stride = dispatch_local_stride;

    dispatch_gpu_nixl_ctx h_dispatch_ctx = {};
    h_dispatch_ctx.local_mvh = nixl_agent_infos[agent_idx].dispatch_local_mvh;
    h_dispatch_ctx.remote_data_mvh = nixl_agent_infos[agent_idx].dispatch_remote_data_mvh;
    h_dispatch_ctx.remote_signal_mvh = nixl_agent_infos[agent_idx].dispatch_remote_signal_mvh;
    h_dispatch_ctx.local_flag_counters = d_dispatch_flag_counters;
    h_dispatch_ctx.num_remote_nodes = num_remote_nodes;
    h_dispatch_ctx.num_channels = gda_num_channels;
    h_dispatch_ctx.rank = rank_uuid;
    h_dispatch_ctx.local_mvh_stride = dispatch_local_stride;
    h_dispatch_ctx.remote_data_mvh_stride = dispatch_remote_stride;

    cudaMalloc(&d_dispatch_nixl_ctx, sizeof(dispatch_gpu_nixl_ctx));
    cudaMemcpy(d_dispatch_nixl_ctx, &h_dispatch_ctx,
               sizeof(dispatch_gpu_nixl_ctx),
               cudaMemcpyHostToDevice);
    NIXL_LOG("    [Rank %d]   Dispatch GPU context built\n", rank_uuid);

    // -- Combine context --
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: Building combine GPU context...\n", rank_uuid);
    cudaMalloc(&d_combine_flag_counters, sizeof(uint64_t) * num_remote_nodes);
    cudaMemset(d_combine_flag_counters, 0, sizeof(uint64_t) * num_remote_nodes);

    int combine_local_stride = 1 + (int)backward_combine;
    int combine_remote_stride = combine_local_stride;

    combine_gpu_nixl_ctx h_combine_ctx = {};
    h_combine_ctx.local_mvh = nixl_agent_infos[agent_idx].combine_local_mvh;
    h_combine_ctx.remote_data_mvh = nixl_agent_infos[agent_idx].combine_remote_data_mvh;
    h_combine_ctx.remote_signal_mvh = nixl_agent_infos[agent_idx].combine_remote_signal_mvh;
    h_combine_ctx.local_flag_counters = d_combine_flag_counters;
    h_combine_ctx.num_remote_nodes = num_remote_nodes;
    h_combine_ctx.num_channels = gda_num_channels;
    h_combine_ctx.rank = rank_uuid;
    h_combine_ctx.local_mvh_stride = combine_local_stride;
    h_combine_ctx.remote_data_mvh_stride = combine_remote_stride;

    cudaMalloc(&d_combine_nixl_ctx, sizeof(combine_gpu_nixl_ctx));
    cudaMemcpy(d_combine_nixl_ctx, &h_combine_ctx,
               sizeof(combine_gpu_nixl_ctx),
               cudaMemcpyHostToDevice);
    NIXL_LOG("    [Rank %d]   Combine GPU context built\n", rank_uuid);

    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: NIXL GPU contexts built (gda_num_channels=%d)\n",
           rank_uuid, gda_num_channels);
}

#define NIXL_REGISTER_BUF(ptr, sz, name) do { \
    if ((ptr) && (sz) > 0) { \
        buffer_count++; \
        nixlBlobDesc desc((uintptr_t)(ptr), (sz), local_device_id, (name)); \
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc); \
        reg_dlist.addDesc(desc); \
    } \
} while (0)

void HybridEP_NIXLConnector::_register_buffers_with_agents() {
    NIXL_LOG("    [Rank %d] _register_buffers_with_agents\n", rank_uuid);
    int agent_idx = 0;
    int buffer_count = 0;
    nixl_reg_dlist_t reg_dlist(VRAM_SEG);

    NIXL_REGISTER_BUF(dispatch_buf->attn_input_token, dispatch_buf->attn_input_token_sz, "attn_input_token");
    NIXL_REGISTER_BUF(dispatch_buf->attn_input_prob, dispatch_buf->attn_input_prob_sz, "attn_input_prob");
    NIXL_REGISTER_BUF(dispatch_buf->attn_input_scaling_factor, dispatch_buf->attn_input_scaling_factor_sz, "attn_input_token_scaling_factor");
    NIXL_REGISTER_BUF(dispatch_buf->rdma_inter_node_group_token, dispatch_buf->rdma_inter_node_group_token_sz, "rdma_inter_node_group_token");
    NIXL_REGISTER_BUF(dispatch_buf->rdma_inter_node_group_flags, dispatch_buf->rdma_inter_node_group_flags_sz, "rdma_inter_node_group_flags");
    NIXL_REGISTER_BUF(dispatch_buf->rdma_inter_node_group_prob, dispatch_buf->rdma_inter_node_group_prob_sz, "rdma_inter_node_group_prob");
    NIXL_REGISTER_BUF(dispatch_buf->rdma_inter_node_group_scaling_factor, dispatch_buf->rdma_inter_node_group_scaling_factor_sz, "rdma_inter_node_group_scaling_factor");
    NIXL_REGISTER_BUF(combine_buf->rdma_intra_node_red_token, combine_buf->rdma_intra_node_red_token_sz, "rdma_intra_node_red_token");
    NIXL_REGISTER_BUF(combine_buf->rdma_intra_node_red_prob, combine_buf->rdma_intra_node_red_prob_sz, "rdma_intra_node_red_prob");
    NIXL_REGISTER_BUF(combine_buf->rdma_inter_node_group_token, combine_buf->rdma_inter_node_group_token_sz, "combine_rdma_inter_node_group_token");
    NIXL_REGISTER_BUF(combine_buf->rdma_inter_node_group_flags, combine_buf->rdma_inter_node_group_flags_sz, "combine_rdma_inter_node_group_flags");
    NIXL_REGISTER_BUF(combine_buf->rdma_inter_node_group_prob, combine_buf->rdma_inter_node_group_prob_sz, "combine_rdma_inter_node_group_prob");

#undef NIXL_REGISTER_BUF

    NIXL_LOG("    [Rank %d] _register_buffers_with_agents: registering %d buffers\n", rank_uuid, buffer_count);
    nixl_status_t status = nixl_agent_infos[agent_idx].agent->registerMem(reg_dlist);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: registerMem failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    status = nixl_agent_infos[agent_idx].agent->sendLocalMD(&nixl_agent_infos[agent_idx].extra_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error(
            "HybridEP NIXL: sendLocalMD failed: " +
            nixlEnumStrings::statusStr(status) + " (code " +
            std::to_string(static_cast<int>(status)) + ").");
    }

    NIXL_LOG("    [Rank %d] _register_buffers_with_agents: done (%d buffers)\n", rank_uuid, buffer_count);
}

}  // namespace hybrid_ep

#endif  // USE_NIXL
