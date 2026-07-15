/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See NCCL_LICENSE.txt for license information
 ************************************************************************/

 #pragma once

 #ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
 #if defined(__x86_64__)
 #include <cpuid.h>
 #endif
 #include <ctype.h>
 #include <float.h>
 #include <infiniband/verbs.h>
 #include <sched.h>
 #include <string.h>
 #include <unistd.h>
 #include <cassert>
 #include <vector>
 #include "nvml.h"
 #include "utils.cuh"
 
 namespace hybrid_ep {
   namespace {
   static constexpr char     HOSTID_FILE[] = "/proc/sys/kernel/random/boot_id";
   static constexpr size_t   BUSID_SIZE = sizeof("0000:00:00.0");
   static constexpr size_t   BUSID_REDUCED_SIZE = sizeof("0000:00");
   static constexpr size_t   CPU_SET_N_U32 = sizeof(cpu_set_t) / sizeof(uint32_t);
   static constexpr size_t   MAX_STR_LEN = 255;
   static constexpr size_t   MAX_ATTR_COUNT = 16;
   static constexpr size_t   MAX_SUBS = 128;
   static constexpr size_t   MAXCHANNELS = 32;
   static constexpr uint32_t NODE_TYPE_NONE = 0;
   static constexpr uint32_t NODE_TYPE_OPEN = 1;
   static constexpr uint32_t NODE_TYPE_CLOSE = 2;
   static constexpr uint32_t NODE_TYPE_SINGLE = 3;
   static constexpr uint32_t GPU = 0;
   static constexpr uint32_t PCI = 1;
   static constexpr uint32_t NVS = 2;
   static constexpr uint32_t CPU = 3; // Actually NUMA domains
   static constexpr uint32_t NIC = 4;
   static constexpr uint32_t NET = 5;
   static constexpr uint32_t NCCL_TOPO_NODE_TYPES = 6;
   static constexpr size_t   NCCL_TOPO_XML_MAX_NODES = 256;
   static constexpr size_t   NCCL_GRAPH_XML_MAX_NODES = 4096;
   static constexpr uint32_t NCCL_TOPO_MAX_LINKS = 128;
   static constexpr uint32_t NCCL_TOPO_MAX_NODES = 576;
   static constexpr uint32_t NCCL_TOPO_MAX_HOPS = NCCL_TOPO_MAX_NODES * NCCL_TOPO_NODE_TYPES;
 
   static constexpr uint32_t LINK_LOC = 0;
   static constexpr uint32_t LINK_NVL = 1;
   // Skipping 2 for PATH_NVB
   static constexpr uint32_t LINK_PCI = 3;
   // Skipping 4 for PATH_PXB
   // Skipping 5 for PATH_PXN
   // Skipping 6 for PATH_PHB
   static constexpr uint32_t LINK_SYS = 7;
   static constexpr uint32_t LINK_NET = 8;
 
   // Local (myself)
   static constexpr uint32_t PATH_LOC = 0;
   // Connection traversing NVLink
   static constexpr uint32_t PATH_NVL = 1;
   // Connection through NVLink using an intermediate GPU
   static constexpr uint32_t PATH_NVB = 2;
   // Connection traversing at most a single PCIe bridge
   static constexpr uint32_t PATH_PIX = 3;
   // Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
   static constexpr uint32_t PATH_PXB = 4;
   // Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
   static constexpr uint32_t PATH_PXN = 5;
   // Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
   static constexpr uint32_t PATH_PHB = 6;
   // Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
   static constexpr uint32_t PATH_SYS = 7;
   // Connection through the network
   static constexpr uint32_t PATH_NET = 8;
   // Disconnected
   static constexpr uint32_t PATH_DIS = 9;
 
   static constexpr float LOC_BW = 5000.0;
   static constexpr float SM60_NVLINK_BW = 18.0;
   static constexpr float SM70_NVLINK_BW = 20.0;
   static constexpr float SM80_NVLINK_BW = 20.0;
   static constexpr float SM90_NVLINK_BW = 20.6;
   static constexpr float SM86_NVLINK_BW = 12.0;
   static constexpr float PCI_BW = 12.0;           // PCI Gen3 x16
   static constexpr float QPI_BW = 6.0;
   static constexpr float AMD_BW = 16.0;
   static constexpr float SKL_QPI_BW = 10.0;
   static constexpr float ZPI_BW = 6.0;
   static constexpr float YONGFENG_ZPI_BW = 9.0;
   static constexpr float P9_BW = 32.0;
   static constexpr float ARM_BW = 6.0;
   static constexpr float NET_BW = 12.0;           // 100Gbit
 
   static constexpr uint32_t NCCL_TOPO_CPU_ARCH_X86 = 1;
   static constexpr uint32_t NCCL_TOPO_CPU_ARCH_POWER = 2;
   static constexpr uint32_t NCCL_TOPO_CPU_ARCH_ARM = 3;
   static constexpr uint32_t NCCL_TOPO_CPU_VENDOR_INTEL = 1;
   static constexpr uint32_t NCCL_TOPO_CPU_VENDOR_AMD = 2;
   static constexpr uint32_t NCCL_TOPO_CPU_VENDOR_ZHAOXIN = 3;
   static constexpr uint32_t NCCL_TOPO_CPU_TYPE_BDW = 1;
   static constexpr uint32_t NCCL_TOPO_CPU_TYPE_SKL = 2;
   static constexpr uint32_t NCCL_TOPO_CPU_TYPE_YONGFENG = 1;
 
   static constexpr uint32_t NCCL_TOPO_CPU_INTEL_BDW = 1;
   static constexpr uint32_t NCCL_TOPO_CPU_INTEL_SKL = 2;
 
   static constexpr int32_t  NCCL_TOPO_UNDEF = -1;
 
   static int ibvWidths[] = { 1, 4, 8, 12, 2 };
   static int ibvSpeeds[] = {
     2500,  /* SDR */
     5000,  /* DDR */
     10000, /* QDR */
     10000, /* QDR */
     14000, /* FDR */
     25000, /* EDR */
     50000, /* HDR */
     100000 /* NDR */ };
 
   struct ncclXmlNode {
     char name[MAX_STR_LEN+1];
     struct {
       char key[MAX_STR_LEN+1];
       char value[MAX_STR_LEN+1];
     } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
     int nAttrs;
     int type;
     struct ncclXmlNode* parent;
     struct ncclXmlNode* subs[MAX_SUBS];
     int nSubs;
   };
 
   struct ncclXml {
     int maxIndex, maxNodes;
     struct ncclXmlNode nodes[1];
   };
 
   struct ncclTopoNode;
   struct ncclTopoLink {
     int type;
     float bw;
     struct ncclTopoNode* remNode;
   };
 
   struct ncclTopoLinkList {
     struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
     int count;
     float bw;
     int type;
   };
 
   struct ncclTopoNode {
     int type;
     int64_t id;
     // Type specific data
     union {
       struct {
         int dev; // NVML dev number
         int rank;
         int cudaCompCap;
         int gdrSupport;
       }gpu;
       struct {
         int dev; // Plugin dev number
         uint64_t asic;
         int port;
         float bw;
         float latency;
         int gdrSupport;
         int collSupport;
         int maxChannels;
         int localGpu;
         const char *name;
       }net;
       struct {
         int arch;
         int vendor;
         int model;
         cpu_set_t affinity;
       }cpu;
       struct {
         uint64_t device;
       }pci;
     };
     int nlinks;
     struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
     // Pre-computed paths to GPUs and NICs
     struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
     // Used during search
     uint64_t used;
   };
 
   struct ncclTopoNodeList {
     struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
     int count;
   };
 
   struct ncclTopoNodeSet {
     int count;
     struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
   };
 
   struct ncclTopoSystem {
     int systemId;
     uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
     int nHosts;
     struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
     float maxBw;
     float totalBw;
   };
 
   struct kvDict {
     const char* str;
     int value;
   };
 
   uint64_t NCCL_TOPO_ID_SYSTEM_ID(uint64_t id) {return id >> 56;}
   uint64_t NCCL_TOPO_ID(int systemid, int localid) {return ((int64_t)systemid << 56) + localid;}
   struct kvDict kvDictCpuArch[] = { { "x86_64", NCCL_TOPO_CPU_ARCH_X86 }, { "arm64", NCCL_TOPO_CPU_ARCH_ARM }, { "ppc64", NCCL_TOPO_CPU_ARCH_POWER }, { NULL, 0 } };
   struct kvDict kvDictCpuVendor[] = { { "GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL }, { "AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD }, { "CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { "  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { NULL, 0 } };
   struct kvDict kvDictPciClass[] = { { "0x060400", PCI }, { "0x068000", NVS }, { "0x068001", CPU }, { "0x03", GPU }, { "0x02", NIC }, { NULL, PCI /* Default fallback value */ } };
   struct kvDict kvDictPciGen[] = {
     { "2.5 GT/s", 15 }, { "5 GT/s", 30 }, { "8 GT/s", 60 }, { "16 GT/s", 120 }, { "32 GT/s", 240 }, /* Kernel 5.6 and earlier */
     { "2.5 GT/s PCIe", 15 }, { "5.0 GT/s PCIe", 30 }, { "8.0 GT/s PCIe", 60 }, { "16.0 GT/s PCIe", 120 }, { "32.0 GT/s PCIe", 240 }, { "64.0 GT/s PCIe", 480 },
     { NULL, 60 /* Default fallback */ } }; // x100 Mbps per lane
 
   ncclResult_t kvConvertToInt(const char* str, int* value, struct kvDict* dict) {
     struct kvDict* d = dict;
     while (d->str) {
       if (strncmp(str, d->str, strlen(d->str)) == 0) {
         *value = d->value;
         return ncclSuccess;
       }
       d++;
     }
     // INFO(NCCL_GRAPH, "KV Convert to int : could not find value of '%s' in dictionary, falling back to %d", str, d->value);
     *value = d->value;
     return ncclSuccess;
   }
 
   int firstBitSet(int val, int max) {
     int i = 0;
     while (i<max && ((val & (1<<i)) == 0)) i++;
     return i;
   }
 
   int ncclIbWidth(int width) {
     return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
   }
 
   int ncclIbSpeed(int speed) {
     return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
   }
 
   uint64_t getHash(const char* string, int n) {
     // Based on DJB2a, result = result * 33 ^ char
     uint64_t result = 5381;
     for (int c = 0; c < n; c++) {
       result = ((result << 5) + result) ^ string[c];
     }
     return result;
   }
 
   ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
     if (gethostname(hostname, maxlen) != 0) {
       strncpy(hostname, "unknown", maxlen);
       return ncclSystemError;
     }
     int i = 0;
     while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
     hostname[i] = '\0';
     return ncclSuccess;
   }
 
   uint64_t getHostHash(void) {
     char hostHash[1024];
     memset(hostHash, 0, sizeof(hostHash));
     // const char *hostId;
     // Fall back is the full hostname if something fails
     (void) getHostName(hostHash, sizeof(hostHash), '\0');
     int offset = strlen(hostHash);
     // if ((hostId = ncclGetEnv("NCCL_HOSTID")) != NULL) {
     //   INFO(NCCL_ENV, "NCCL_HOSTID set by environment to %s", hostId);
     //   strncpy(hostHash, hostId, sizeof(hostHash));
     // } else {
       FILE *file = fopen(HOSTID_FILE, "r");
       if (file != NULL) {
         char *p;
         if (fscanf(file, "%ms", &p) == 1) {
           strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
           free(p);
         }
       }
       fclose(file);
     // }
     // Make sure the string is terminated
     hostHash[sizeof(hostHash)-1]='\0';
     return getHash(hostHash, strlen(hostHash));
   }
 
   ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id, struct ncclTopoLinkList** path) {
     for (int i=0; i<system->nodes[t].count; i++) {
       if (system->nodes[t].nodes[i].id == id) {
         *path = node->paths[t]+i;
         return ncclSuccess;
       }
     }
     // WARN("Could not find node of type %d id %lx", t, id);
     return ncclInternalError;
   }
 
   int isHex(char c) {
     return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
   }
 
   int hexToInt(char c) {
     int v = c - '0';
     if (v < 0) return -1;
     if (v > 9) v = 10 + c - 'a';
     if ((v < 0) || (v > 15)) return -1;
     return v;
   }
 
   int checkBDFFormat(char* bdf) {
     if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.') return 0;
     if (isHex(bdf[0]) == 0 || isHex(bdf[1]) == 0 || isHex(bdf[2]) == 0 || isHex(bdf[3]) == 0 ||
         isHex(bdf[5]) == 0 || isHex(bdf[6]) == 0 || isHex(bdf[8]) == 0 || isHex(bdf[9]) == 0 ||
         isHex(bdf[11]) == 0) return 0;
     return 1;
   }
 
   void memcpylower(char* dst, const char* src, const size_t size) {
     for (int i=0; i<size; i++) dst[i] = tolower(src[i]);
   }
 
   ncclResult_t getPciPath(const char* busId, char** path) {
     char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
     memcpylower(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
     memcpylower(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
     *path = realpath(busPath, NULL);
     if (*path == NULL) {
       // WARN("Could not find real path of %s", busPath);
       return ncclSystemError;
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclIbGetPciPath(char* devName, char** path) {
     char devicePath[PATH_MAX];
     snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
     char* p = realpath(devicePath, NULL);
     // if (p == NULL) {
     //   WARN("Could not find real path of %s (%s)", devName, devicePath);
     // } else {
     //   // Merge multi-port NICs into the same PCI device
     //   p[strlen(p)-1] = '0';
     //   // Also merge virtual functions (VF) into the same device
     //   if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
     //   // And keep the real port aside (the ibv port is always 1 on recent cards)
     //   *realPort = 0;
     //   for (int d=0; d<ncclNIbDevs; d++) {
     //     if (strcmp(p, ncclIbDevs[d].pciPath) == 0) (*realPort)++;
     //   }
     // }
     if (p != NULL) {
       p[strlen(p)-1] = '0';
     }
     *path = p;
     return ncclSuccess;
   }
 
   ncclResult_t ncclStrToCpuset(const char* str, cpu_set_t* mask) {
     uint32_t cpumasks[CPU_SET_N_U32];
     int m = CPU_SET_N_U32-1;
     cpumasks[m] = 0;
     for (int o=0; o<strlen(str); o++) {
       char c = str[o];
       if (c == ',') {
         m--;
         cpumasks[m] = 0;
       } else {
         int v = hexToInt(c);
         if (v == -1) break;
         cpumasks[m] <<= 4;
         cpumasks[m] += v;
       }
     }
     // Copy cpumasks to mask
     for (int a=0; m<CPU_SET_N_U32; a++,m++) {
       memcpy(((uint32_t*)mask)+a, cpumasks+m, sizeof(uint32_t));
     }
     return ncclSuccess;
   }
 
   ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
     char hexStr[17];  // Longest possible int64 hex string + null terminator.
     int hexOffset = 0;
     for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
       char c = busId[i];
       if (c == '.' || c == ':') continue;
       if ((c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'F') ||
           (c >= 'a' && c <= 'f')) {
         hexStr[hexOffset++] = busId[i];
       } else break;
     }
     hexStr[hexOffset] = '\0';
     *id = strtol(hexStr, NULL, 16);
     return ncclSuccess;
   }
 
   size_t xmlMemSize(int maxNodes) {
     return offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode)*maxNodes;
   }
 
   ncclResult_t xmlAddNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName, struct ncclXmlNode** sub) {
     if (xml->maxIndex == xml->maxNodes) {
       // WARN("Error : too many XML nodes (max %d)", xml->maxNodes);
       return ncclInternalError;
     }
     struct ncclXmlNode* s = xml->nodes+xml->maxIndex++;
     s->nSubs = 0;
     s->nAttrs = 0;
     *sub = s;
     s->parent = parent;
     if (parent) {
       if (parent->nSubs == MAX_SUBS) {
         // WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
         return ncclInternalError;
       }
       parent->subs[parent->nSubs++] = s;
     }
     strncpy(s->name, subName, MAX_STR_LEN);
     s->name[MAX_STR_LEN] = '\0';
     return ncclSuccess;
   }
 
   ncclResult_t xmlAlloc(struct ncclXml** xml, int maxNodes) {
     char* mem;
     NCCL_CHECK(ncclCalloc(&mem, xmlMemSize(maxNodes)));
     *xml = (struct ncclXml*)mem;
     (*xml)->maxNodes = maxNodes;
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttrIndex(struct ncclXmlNode* node, const char* attrName, int* index) {
     *index = -1;
     const int nAttrs = node->nAttrs;
     for (int a=0; a<nAttrs; a++) {
       if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
         *index = a;
         return ncclSuccess;
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttr(struct ncclXmlNode* node, const char* attrName, const char** value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     *value = index == -1 ? NULL : node->attrs[index].value;
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttrStr(struct ncclXmlNode* node, const char* attrName, const char** value) {
     NCCL_CHECK(xmlGetAttr(node, attrName, value));
     if (*value == NULL) {
       // WARN("Attribute %s of node %s not found", attrName, node->name);
       return ncclInternalError;
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttrInt(struct ncclXmlNode* node, const char* attrName, int* value) {
     const char* str;
     NCCL_CHECK(xmlGetAttrStr(node, attrName, &str));
     *value = strtol(str, NULL, 0);
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttrIntDefault(struct ncclXmlNode* node, const char* attrName, int* value, int defaultValue) {
     const char* str;
     NCCL_CHECK(xmlGetAttr(node, attrName, &str));
     *value = str ? strtol(str, NULL, 0) : defaultValue;
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttrLong(struct ncclXmlNode* node, const char* attrName, int64_t* value) {
     const char* str;
     NCCL_CHECK(xmlGetAttrStr(node, attrName, &str));
     *value = strtol(str, NULL, 0);
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetAttrFloat(struct ncclXmlNode* node, const char* attrName, float* value) {
     const char* str;
     NCCL_CHECK(xmlGetAttrStr(node, attrName, &str));
     *value = strtof(str, NULL);
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetSub(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub) {
     *sub = NULL;
     for (int s=0; s<node->nSubs; s++) {
       if (strcmp(node->subs[s]->name, subName) == 0) {
         *sub = node->subs[s];
         return ncclSuccess;
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlGetSubKv(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub, const char* attrName, const char* attrValue) {
     *sub = NULL;
     for (int s=0; s<node->nSubs; s++) {
       struct ncclXmlNode* subNode = node->subs[s];
       if (strcmp(subNode->name, subName) == 0) {
         const char* value;
         NCCL_CHECK(xmlGetAttr(subNode, attrName, &value));
         if (value && strcmp(value, attrValue) == 0) {
           *sub = node->subs[s];
           return ncclSuccess;
         }
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlFindNextTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode* prev, struct ncclXmlNode** node) {
     *node = NULL;
     for (int i=prev-xml->nodes+1; i<xml->maxIndex; i++) {
       struct ncclXmlNode* n = xml->nodes+i;
       if (strcmp(n->name, tagName) == 0) {
         *node = n;
         return ncclSuccess;
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlFindTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node) {
     *node = NULL;
     for (int i=0; i<xml->maxIndex; i++) {
       struct ncclXmlNode* n = xml->nodes+i;
       if (strcmp(n->name, tagName) == 0) {
         *node = n;
         return ncclSuccess;
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlFindTagKv(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node, const char* attrName, const char* attrValue) {
     *node = NULL;
     for (int i=0; i<xml->maxIndex; i++) {
       struct ncclXmlNode* n = xml->nodes+i;
       if (strcmp(n->name, tagName) == 0) {
         const char* value;
         NCCL_CHECK(xmlGetAttr(n, attrName, &value));
         if (value && strcmp(value, attrValue) == 0) {
           *node = n;
           return ncclSuccess;
         }
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index == -1) {
       index = node->nAttrs++;
       strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
       snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index == -1) {
       index = node->nAttrs++;
       strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
       snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index == -1) {
       index = node->nAttrs++;
       strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
       snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
     }
     return ncclSuccess;
   }
 
   ncclResult_t xmlSetAttr(struct ncclXmlNode* node, const char* attrName, const char* value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index == -1) {
       index = node->nAttrs++;
       strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
       node->attrs[index].key[MAX_STR_LEN] = '\0';
     }
     strncpy(node->attrs[index].value, value, MAX_STR_LEN);
     node->attrs[index].value[MAX_STR_LEN] = '\0';
     return ncclSuccess;
   }
 
   ncclResult_t xmlSetAttrIfUnset(struct ncclXmlNode* node, const char* attrName, const char* value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index != -1) return ncclSuccess;
     index = node->nAttrs++;
     strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
     node->attrs[index].key[MAX_STR_LEN] = '\0';
     strncpy(node->attrs[index].value, value, MAX_STR_LEN);
     node->attrs[index].value[MAX_STR_LEN] = '\0';
     return ncclSuccess;
   }
 
   ncclResult_t xmlSetAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index == -1) {
       index = node->nAttrs++;
       strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
       node->attrs[index].key[MAX_STR_LEN] = '\0';
     }
     snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
     node->attrs[index].value[MAX_STR_LEN] = '\0';
     return ncclSuccess;
   }
 
   ncclResult_t xmlSetAttrLong(struct ncclXmlNode* node, const char* attrName, const int64_t value) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(node, attrName, &index));
     if (index == -1) {
       index = node->nAttrs++;
       strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
       node->attrs[index].key[MAX_STR_LEN] = '\0';
     }
     snprintf(node->attrs[index].value, MAX_STR_LEN, "%#lx", value);
     node->attrs[index].value[MAX_STR_LEN] = '\0';
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetStrFromSys(const char* path, const char* fileName, char* strValue) {
     char filePath[PATH_MAX];
     sprintf(filePath, "%s/%s", path, fileName);
     int offset = 0;
     FILE* file;
     if ((file = fopen(filePath, "r")) != NULL) {
       while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
         int len = fread(strValue+offset, 1, MAX_STR_LEN-offset, file);
         offset += len;
       }
       fclose(file);
     }
     if (offset == 0) {
       strValue[0] = '\0';
       // INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
     } else {
       strValue[offset-1] = '\0';
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
     char subSysPath[PATH_MAX];
     sprintf(subSysPath, "%s/subsystem", sysPath);
     char* path = realpath(subSysPath, NULL);
     if (path == NULL) {
       subSys[0] = '\0';
     } else {
       int offset;
       for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
       strcpy(subSys, path+offset+1);
       free(path);
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName, const char* attrName) {
     char strValue[MAX_STR_LEN];
     NCCL_CHECK(ncclTopoGetStrFromSys(path, fileName, strValue));
     if (strValue[0] != '\0') { NCCL_CHECK(xmlSetAttr(pciNode, attrName, strValue)); }
     // TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
     int index;
     NCCL_CHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
     if (index == -1) {
       const char* numaId;
       NCCL_CHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
       if (numaId == NULL) {
         // WARN("GetXmlFromCpu : could not find CPU numa ID.");
         return ncclInternalError;
       }
       // Set affinity
       char cpumaskPath[] = "/sys/devices/system/node/node0000";
       sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
       NCCL_CHECK(ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
     }
     NCCL_CHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
     if (index == -1) {
       // Fill CPU type / vendor / model
   #if defined(__PPC__)
       NCCL_CHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
   #elif defined(__aarch64__)
       NCCL_CHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
   #elif defined(__x86_64__)
       NCCL_CHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
   #endif
     }
 
   #if defined(__x86_64__)
     NCCL_CHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
     if (index == -1) {
       union {
         struct {
           // CPUID 0 String register order
           uint32_t ebx;
           uint32_t edx;
           uint32_t ecx;
         };
         char vendor[12];
       } cpuid0;
 
       [[maybe_unused]] unsigned unused;
       __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
       char vendor[13];
       strncpy(vendor, cpuid0.vendor, 12);
       vendor[12] = '\0';
       NCCL_CHECK(xmlSetAttr(cpuNode, "vendor", vendor));
     }
     NCCL_CHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
     if (index == -1) {
       union {
         struct {
           unsigned steppingId:4;
           unsigned modelId:4;
           unsigned familyId:4;
           unsigned processorType:2;
           unsigned resv0:2;
           unsigned extModelId:4;
           unsigned extFamilyId:8;
           unsigned resv1:4;
         };
         uint32_t val;
       } cpuid1;
       [[maybe_unused]] unsigned unused;
       __cpuid(1, cpuid1.val, unused, unused, unused);
       int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
       int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
       NCCL_CHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
       NCCL_CHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
     }
   #endif
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, nvmlDevice_t nvmlDev, struct ncclXml* xml, struct ncclXmlNode** gpuNodeRet) {
     struct ncclXmlNode* gpuNode = NULL;
     NCCL_CHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
     if (gpuNode == NULL) NCCL_CHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));
     int index = -1;
     int dev = -1;
     NCCL_CHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
     if (index == -1) {
       CALL_CHECK(nvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev));
       NCCL_CHECK(xmlSetAttrInt(gpuNode, "dev", dev));
     }
     NCCL_CHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
     if (dev == -1) { *gpuNodeRet = NULL; return ncclSuccess; }
     NCCL_CHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
     if (index == -1) {
       int cudaMajor, cudaMinor;
       if (nvmlDev == NULL) {
         cudaDeviceProp devProp;
         CUDA_CHECK(cudaGetDeviceProperties(&devProp, dev));
         cudaMajor = devProp.major; cudaMinor = devProp.minor;
       } else {
         CALL_CHECK(nvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
       }
       NCCL_CHECK(xmlSetAttrInt(gpuNode, "sm", cudaMajor*10+cudaMinor));
     }
     int sm;
     NCCL_CHECK(xmlGetAttrInt(gpuNode, "sm", &sm));
     struct ncclXmlNode* nvlNode = NULL;
     NCCL_CHECK(xmlGetSub(gpuNode, "nvlink", &nvlNode));
     if (nvlNode == NULL) {
       // NVML NVLink detection
       int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : (sm < 80) ? 6 : (sm < 90) ? 12 : 18;
       if (maxNvLinks > 0 && nvmlDev == NULL) {
         // WARN("No NVML device handle. Skipping nvlink detection.");
         maxNvLinks = 0;
       }
       for (int l=0; l<maxNvLinks; ++l) {
         // Check whether we can use this NVLink for P2P
         unsigned canP2P;
         if ((nvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != NVML_SUCCESS) || !canP2P) continue;
         // Make sure the Nvlink is up. The previous call should have trained the link.
         nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
   #if CUDART_VERSION >= 11080
         if (sm >= 90) {
           nvmlFieldValue_t fv;
           fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
           fv.scopeId = l;
           // fv.value will contain NV_FEATURE_ENABLED or NV_FEATURE_DISABLED
           if ((nvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == NVML_SUCCESS) && (fv.nvmlReturn == NVML_SUCCESS))
             isActive = (nvmlEnableState_t) fv.value.uiVal;
         } else /* FALLTHRU to GetNvLinkState if before SM90 */
   #endif
         {
           (void) nvmlDeviceGetNvLinkState(nvmlDev, l, &isActive);
         }
         if (isActive != NVML_FEATURE_ENABLED) continue;
         // Try to figure out what's on the other side of the NVLink
         nvmlPciInfo_t remoteProc;
         if (nvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != NVML_SUCCESS) continue;
         // Make a lower case copy of the bus ID for calling ncclDeviceType
         // PCI system path is in lower case
         char* p = remoteProc.busId;
         char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
         for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
           lowerId[c] = tolower(p[c]);
           if (p[c] == 0) break;
         }
         NCCL_CHECK(xmlGetSubKv(gpuNode, "nvlink", &nvlNode, "target", lowerId));
         if (nvlNode == NULL) {
           NCCL_CHECK(xmlAddNode(xml, gpuNode, "nvlink", &nvlNode));
           NCCL_CHECK(xmlSetAttr(nvlNode, "target", lowerId));
           NCCL_CHECK(xmlSetAttrInt(nvlNode, "count", 1));
         } else {
           int count;
           NCCL_CHECK(xmlGetAttrInt(nvlNode, "count", &count));
           NCCL_CHECK(xmlSetAttrInt(nvlNode, "count", count+1));
         }
       }
     }
   #if CUDART_VERSION >= 11080
     struct ncclXmlNode* c2cNode = NULL;
     NCCL_CHECK(xmlGetSub(gpuNode, "c2c", &c2cNode));
     if (c2cNode == NULL) {
         if (sm >= 90) {
           int c2cLinksCount = 0;
           nvmlFieldValue_t fv;
           fv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
           if ((nvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == NVML_SUCCESS) && (fv.nvmlReturn == NVML_SUCCESS)) {
             c2cLinksCount = fv.value.uiVal;
             int bw = 0;
             int count = 0;
             for (int l=0; l<c2cLinksCount; l++) {
               nvmlFieldValue_t fvs[2];
               fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
               fvs[0].scopeId = l;
               fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
               fvs[1].scopeId = l;
               if ((nvmlDeviceGetFieldValues(nvmlDev, 2, fvs) == NVML_SUCCESS) &&
                   (fvs[0].nvmlReturn == NVML_SUCCESS) &&
                   (fvs[0].value.uiVal == 1) &&
                   (fvs[1].nvmlReturn == NVML_SUCCESS)) {
                 bw = fvs[1].value.uiVal;
                 count++;
               }
             }
             if (count > 0) {
               NCCL_CHECK(xmlAddNode(xml, gpuNode, "c2c", &c2cNode));
               NCCL_CHECK(xmlSetAttrInt(c2cNode, "bw", bw));
               NCCL_CHECK(xmlSetAttrInt(c2cNode, "count", count));
             }
           }
         }
     }
   #endif
     // Fill target classes
     for (int s=0; s<gpuNode->nSubs; s++) {
       struct ncclXmlNode* sub = gpuNode->subs[s];
       if (strcmp(sub->name, "nvlink") != 0) continue;
       int index;
       NCCL_CHECK(xmlGetAttrIndex(sub, "tclass", &index));
       if (index == -1) {
         const char* busId;
         NCCL_CHECK(xmlGetAttr(sub, "target", &busId));
         char* path;
         // ncclDebugNoWarn = NCCL_GRAPH;
         getPciPath(busId, &path);
         // ncclDebugNoWarn = 0;
         if (path == NULL || strcmp(busId, "fffffff:ffff:ff") == 0) {
           // Remote NVLink device is not visible inside this VM. Assume NVSwitch.
           NCCL_CHECK(xmlSetAttr(sub, "tclass", "0x068000"));
         } else {
           NCCL_CHECK(ncclTopoSetAttrFromSys(sub, path, "class", "tclass"));
           free(path);
         }
       }
     }
     *gpuNodeRet = gpuNode;
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
     // Fill info, then parent
     const char* busId;
     NCCL_CHECK(xmlGetAttr(pciNode, "busid", &busId));
     char* path = NULL;
     // ncclDebugNoWarn = NCCL_GRAPH;
     getPciPath(busId, &path);
     // ncclDebugNoWarn = 0;
     if (path) {
       NCCL_CHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
     }
     int index;
     // ncclDebugNoWarn = NCCL_GRAPH;
     NCCL_CHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
     if (index == -1) {
       if (path) ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
     }
     NCCL_CHECK(xmlGetAttrIndex(pciNode, "device", &index));
     if (index == -1) {
       if (path) ncclTopoSetAttrFromSys(pciNode, path, "device", "device");
     }
     NCCL_CHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
     if (index == -1) {
       if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor");
     }
     NCCL_CHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
     if (index == -1) {
       if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device");
     }
     // ncclDebugNoWarn = 0;
     NCCL_CHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
     if (index == -1) {
       if (path) {
         char deviceSpeedStr[MAX_STR_LEN];
         float deviceSpeed = FLT_MAX;
         NCCL_CHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));
         sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
         char portSpeedStr[MAX_STR_LEN];
         float portSpeed = FLT_MAX;
         NCCL_CHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
         sscanf(portSpeedStr, "%f GT/s", &portSpeed);
         NCCL_CHECK(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
       } else {
         NCCL_CHECK(xmlSetAttr(pciNode, "link_speed", ""));
       }
     }
     NCCL_CHECK(xmlGetAttrIndex(pciNode, "link_width", &index));
     if (index == -1) {
       if (path) {
         char strValue[MAX_STR_LEN];
         NCCL_CHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
         int deviceWidth = strtol(strValue, NULL, 0);
         NCCL_CHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
         int portWidth = strtol(strValue, NULL, 0);
         NCCL_CHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
       } else {
         NCCL_CHECK(xmlSetAttr(pciNode, "link_width", ""));
       }
     }
     struct ncclXmlNode* parent = pciNode->parent;
     if (parent == NULL) {
       if (path) {
         // Save that for later in case next step is a CPU
         char numaIdStr[MAX_STR_LEN];
         NCCL_CHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));
 
         // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
         // switch, or stop if we reach a CPU root complex.
         int slashCount = 0;
         int parentOffset;
         for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
           if (path[parentOffset] == '/') {
             slashCount++;
             path[parentOffset] = '\0';
             int start = parentOffset - 1;
             while (start>0 && path[start] != '/') start--;
             // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
             if (checkBDFFormat(path+start+1) == 0) {
               // This a CPU root complex. Create a CPU tag and stop there.
               struct ncclXmlNode* topNode;
               NCCL_CHECK(xmlFindTag(xml, "system", &topNode));
               NCCL_CHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
               if (parent == NULL) {
                 NCCL_CHECK(xmlAddNode(xml, topNode, "cpu", &parent));
                 NCCL_CHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
                 NCCL_CHECK(xmlSetAttr(parent, "numaid", numaIdStr));
               }
             } else if (slashCount == 2) {
               // Continue on the upper PCI switch
               for (int i = strlen(path)-1; i>0; i--) {
                 if (path[i] == '/') {
                   NCCL_CHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));
                   if (parent == NULL) {
                     NCCL_CHECK(xmlAddNode(xml, NULL, "pci", &parent));
                     NCCL_CHECK(xmlSetAttr(parent, "busid", path+i+1));
                   }
                   break;
                 }
               }
             }
           }
           if (parent) break;
         }
       } else {
         // No information on /sys, attach GPU to unknown CPU
         NCCL_CHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
         if (parent == NULL) {
           struct ncclXmlNode* topNode;
           NCCL_CHECK(xmlFindTag(xml, "system", &topNode));
           NCCL_CHECK(xmlAddNode(xml, topNode, "cpu", &parent));
           NCCL_CHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
           NCCL_CHECK(xmlSetAttr(parent, "numaid", "-1"));
           NCCL_CHECK(ncclTopoGetXmlFromCpu(parent, xml));
         }
       }
       pciNode->parent = parent;
       // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
       int subIndex = parent->nSubs;
       const char* newBusId;
       NCCL_CHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
       for (int s=0; s<parent->nSubs; s++) {
         const char* busId;
         NCCL_CHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
         if (busId != NULL && strcmp(newBusId, busId) < 0) { subIndex = s; break; }
       }
       if (parent->nSubs == MAX_SUBS) {
         // WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
         return ncclInternalError;
       }
       for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s-1];
       parent->subs[subIndex] = pciNode;
       parent->nSubs++;
     }
     if (strcmp(parent->name, "pci") == 0) {
       NCCL_CHECK(ncclTopoGetXmlFromSys(parent, xml));
     } else if (strcmp(parent->name, "cpu") == 0) {
       NCCL_CHECK(ncclTopoGetXmlFromCpu(parent, xml));
     }
     free(path);
     return ncclSuccess;
   }
 
   ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
     const char* hostHashStr;
     NCCL_CHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
     uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
     int systemId;
     for (systemId=0; systemId<system->nHosts; systemId++) if (system->hostHashes[systemId] == hostHash) break;
     if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;
     *systemIdPtr = systemId;
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                                 int* locals, int* localCount, int* pathType) {
     int minType = PATH_DIS;
     float maxBw = 0;
     int count = 0;
     struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
     if (paths == NULL) { *localCount = 0; return ncclInternalError; }
     for (int i=0; i<system->nodes[resultType].count; i++) {
       if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
         maxBw = paths[i].bw;
         minType = paths[i].type;
         if (pathType) *pathType = minType;
         count = 0;
       }
       if (paths[i].bw == maxBw && paths[i].type == minType) {
         if (count == NCCL_TOPO_MAX_NODES) {
           return ncclInternalError;
         }
         locals[count++] = i;
       }
     }
     *localCount = count;
     // int minType = PATH_DIS;
     // float maxBw = 0;
     // int count = 0;
     // NCCL_CHECK(ncclCalloc(locals, system->nodes[resultType].count));
     // struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
     // for (int i=0; i<system->nodes[resultType].count; i++) {
     //   if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
     //     maxBw = paths[i].bw;
     //     minType = paths[i].type;
     //     if (pathType) *pathType = minType;
     //     count = 0;
     //   }
     //   if (paths[i].bw == maxBw && paths[i].type == minType) (*locals)[count++] = i;
     // }
     // *localCount = count;
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
     *bw = LOC_BW;
     if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
       *bw = P9_BW;
       return ncclSuccess;
     }
     if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
       *bw = ARM_BW;
       return ncclSuccess;
     }
     if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
       *bw = cpu->cpu.model == NCCL_TOPO_CPU_TYPE_SKL ? SKL_QPI_BW : QPI_BW;
     }
     if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
       *bw = AMD_BW;
     }
     if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
       *bw = cpu->cpu.model ==  NCCL_TOPO_CPU_TYPE_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
     // Aggregate links into higher bw for NVLink
     struct ncclTopoLink* link;
     for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
       if (link->remNode == remNode && link->type == type) break;
     }
     if (link - node->links == NCCL_TOPO_MAX_LINKS) {
       // WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
       return ncclInternalError;
     }
     if (link->remNode == NULL) node->nlinks++;
     link->type = type;
     link->remNode = remNode;
     link->bw += bw;
 
     // Sort links in BW descending order
     struct ncclTopoLink linkSave;
     memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
     while (link != node->links) {
       if ((link-1)->bw >= linkSave.bw) break;
       memcpy(link, link-1, sizeof(struct ncclTopoLink));
       link--;
     }
     memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
     // And connect all CPU nodes together
     for (int n=0; n<system->nodes[CPU].count; n++) {
       struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes+n;
       for (int p=0; p<system->nodes[CPU].count; p++) {
         struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes+p;
         if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
         float bw;
         NCCL_CHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
         NCCL_CHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
     if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
       // WARN("Error : tried to create too many nodes of type %d", type);
       return ncclInternalError;
     }
     struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
     system->nodes[type].count++;
     n->type = type;
     n->id = id;
     if (type == GPU) {
       // Create link to itself (used in some corner cases)
       n->nlinks=1;
       n->links[0].type = LINK_LOC;
       n->links[0].remNode = n;
       n->links[0].bw = LOC_BW;
       n->gpu.dev = NCCL_TOPO_UNDEF;
       n->gpu.rank = NCCL_TOPO_UNDEF;
       n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
     } else if (type == CPU) {
       n->cpu.arch = NCCL_TOPO_UNDEF;
       n->cpu.vendor = NCCL_TOPO_UNDEF;
       n->cpu.model = NCCL_TOPO_UNDEF;
     } else if (type == NET) {
       n->net.asic = 0ULL;
       n->net.port = NCCL_TOPO_UNDEF;
       n->net.bw = 0.0;
       n->net.latency = 0.0;
     }
     *node = n;
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
     for (int i=0; i<system->nodes[type].count; i++) {
       if (system->nodes[type].nodes[i].id == id) {
         *node = system->nodes[type].nodes+i;
         return ncclSuccess;
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
     NCCL_CHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
     if (*pciNode == NULL) {
       NCCL_CHECK(xmlAddNode(xml, NULL, "pci", pciNode));
       NCCL_CHECK(xmlSetAttr(*pciNode, "busid", busId));
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
     NCCL_CHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
     NCCL_CHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
     NCCL_CHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
     NCCL_CHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
     // Do not go any further, nvlinks will be added in a second pass
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
     int dev;
     NCCL_CHECK(xmlGetAttrInt(xmlNet, "dev", &dev));
     struct ncclTopoNode* net;
     NCCL_CHECK(ncclTopoCreateNode(system, &net, NET, NCCL_TOPO_ID(systemId, dev)));
     net->net.dev = dev;
     const char* str;
     NCCL_CHECK(xmlGetAttr(xmlNet, "guid", &str));
     if (str) sscanf(str, "0x%lx", &net->net.asic);
     else net->net.asic = dev;
     // ncclDebugNoWarn = NCCL_GRAPH;
     int mbps;
     NCCL_CHECK(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0));
     if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
     net->net.bw = mbps / 8000.0;
     if (xmlGetAttrFloat(xmlNet, "latency", &net->net.latency) != ncclSuccess) net->net.latency = 0;
     NCCL_CHECK(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0));
     NCCL_CHECK(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0));
     NCCL_CHECK(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS));
     NCCL_CHECK(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0));
     NCCL_CHECK(xmlGetAttrStr(xmlNet, "name", &net->net.name));
     // ncclDebugNoWarn = 0;
     NCCL_CHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
     NCCL_CHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
     for (int s=0; s<xmlNic->nSubs; s++) {
       struct ncclXmlNode* xmlNet = xmlNic->subs[s];
       if (strcmp(xmlNet->name, "net") != 0) continue;
       int index;
       NCCL_CHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
       if (index == -1) continue;
       NCCL_CHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId) {
     const char* str;
     int type;
     NCCL_CHECK(xmlGetAttrStr(xmlPci, "class", &str));
     NCCL_CHECK(kvConvertToInt(str, &type, kvDictPciClass));
     int64_t busId;
     NCCL_CHECK(xmlGetAttrStr(xmlPci, "busid", &str));
     NCCL_CHECK(busIdToInt64(str, &busId));
     struct ncclTopoNode* node = NULL;
     struct ncclXmlNode* xmlGpu = NULL;
     NCCL_CHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
     if (xmlGpu != NULL) {
       type = GPU;
       int index;
       NCCL_CHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
       if (index == -1) return ncclSuccess;
       NCCL_CHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
       NCCL_CHECK(ncclTopoAddGpu(xmlGpu, system, node));
     }
     struct ncclXmlNode* xmlNic = NULL;
     NCCL_CHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
     if (xmlNic != NULL) {
       type = NIC;
       // Ignore sub device ID and merge multi-port NICs into one PCI device.
       busId &= 0xfffffffffffffff0;
       struct ncclTopoNode* nicNode = NULL;
       int64_t id = NCCL_TOPO_ID(systemId, busId);
       NCCL_CHECK(ncclTopoGetNode(system, &nicNode, type, id));
       if (nicNode == NULL) {
         NCCL_CHECK(ncclTopoCreateNode(system, &nicNode, type, id));
         node = nicNode; // Connect it to parent later on
       }
       NCCL_CHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
     } else if (type == PCI) {
       NCCL_CHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
       NCCL_CHECK(xmlGetAttr(xmlPci, "vendor", &str));
       if (str) node->pci.device += strtol(str, NULL, 0) << 48;
       NCCL_CHECK(xmlGetAttr(xmlPci, "device", &str));
       if (str) node->pci.device += strtol(str, NULL, 0) << 32;
       NCCL_CHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
       if (str) node->pci.device += strtol(str, NULL, 0) << 16;
       NCCL_CHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
       if (str) node->pci.device += strtol(str, NULL, 0);
       for (int s=0; s<xmlPci->nSubs; s++) {
         struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
         NCCL_CHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId));
       }
     }
     if (node) {
       int width, speed;
       NCCL_CHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
       NCCL_CHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));
 
       // Manage cases where speed was not indicated in /sys
       if (width == 0) width = 16;
       NCCL_CHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)
 
       NCCL_CHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, width*speed/80.0));
       NCCL_CHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, width*speed/80.0));
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
     int numaId;
     NCCL_CHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
     int systemId;
     NCCL_CHECK(ncclGetSystemId(system, xmlCpu, &systemId));
     struct ncclTopoNode* cpu;
     NCCL_CHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
     const char* str;
     NCCL_CHECK(xmlGetAttr(xmlCpu, "affinity", &str));
     if (str != NULL) {
       NCCL_CHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
     }
 
     NCCL_CHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
     NCCL_CHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
     if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
       NCCL_CHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
       NCCL_CHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
       if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
         int familyId, modelId;
         NCCL_CHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
         NCCL_CHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
         cpu->cpu.model = (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_TYPE_SKL : NCCL_TOPO_CPU_INTEL_BDW;
       } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
         int familyId, modelId;
         NCCL_CHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
         NCCL_CHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
         if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_TYPE_YONGFENG;
       }
     }
     for (int s=0; s<xmlCpu->nSubs; s++) {
       struct ncclXmlNode* node = xmlCpu->subs[s];
       if (strcmp(node->name, "pci") == 0) NCCL_CHECK(ncclTopoAddPci(node, system, cpu, systemId));
       if (strcmp(node->name, "nic") == 0) {
         struct ncclTopoNode* nic = NULL;
         NCCL_CHECK(ncclTopoGetNode(system, &nic, NIC, 0));
         if (nic == NULL) {
           NCCL_CHECK(ncclTopoCreateNode(system, &nic, NIC, NCCL_TOPO_ID(systemId, 0)));
           NCCL_CHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
           NCCL_CHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
         }
         NCCL_CHECK(ncclTopoAddNic(node, system, nic, systemId));
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
     struct ncclXmlNode* node;
     NCCL_CHECK(ncclTopoGetPciNode(xml, busId, &node));
     NCCL_CHECK(xmlSetAttrIfUnset(node, "class", "0x03"));
     NCCL_CHECK(ncclTopoGetXmlFromSys(node, xml));
     nvmlDevice_t nvmlDev;
     CALL_CHECK(nvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
     NCCL_CHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode) {
     NCCL_CHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
     if (*netNode != NULL) return ncclSuccess;
     const char* pciSysPath = pciPath;
     if (pciSysPath) {
       char subSystem[PATH_MAX];
       NCCL_CHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
       // This is not a PCI device (virtual, usb, ...).
       if (strcmp(subSystem, "pci") != 0) {
         // INFO(NCCL_GRAPH, "Topology detection: network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSystem);
         pciSysPath = NULL;
       }
     }
     struct ncclXmlNode* parent = NULL;
     if (pciSysPath) {
       int offset;
       for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
       char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
       strcpy(busId, pciSysPath+offset+1);
       NCCL_CHECK(ncclTopoGetPciNode(xml, busId, &parent));
       NCCL_CHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));
       NCCL_CHECK(ncclTopoGetXmlFromSys(parent, xml));
     } else {
       // Virtual NIC, no PCI device, attach to first CPU
       NCCL_CHECK(xmlFindTag(xml, "cpu", &parent));
     }
     struct ncclXmlNode* nicNode = NULL;
     NCCL_CHECK(xmlGetSub(parent, "nic", &nicNode));
     if (nicNode == NULL) {
       NCCL_CHECK(xmlAddNode(xml, parent, "nic", &nicNode));
     }
     // We know that this net does not exist yet (we searched for it at the
     // beginning of this function), so we can add it.
     NCCL_CHECK(xmlAddNode(xml, nicNode, "net", netNode));
     NCCL_CHECK(xmlSetAttr(*netNode, "name", netName));
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
     if (baseNode->paths[baseNode->type] == NULL) {
       NCCL_CHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
     }
     // breadth-first search to set all paths to that node in the system
     struct ncclTopoNodeList nodeList;
     struct ncclTopoNodeList nextNodeList;
     nodeList.count = 1; nodeList.list[0] = baseNode;
     nextNodeList.count = 0;
     struct ncclTopoLinkList* basePath;
     NCCL_CHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
     basePath->count = 0;
     basePath->bw = LOC_BW;
     basePath->type = PATH_LOC;
 
     while (nodeList.count) {
       nextNodeList.count = 0;
       for (int n=0; n<nodeList.count; n++) {
         struct ncclTopoNode* node = nodeList.list[n];
         struct ncclTopoLinkList* path;
         NCCL_CHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
         for (int l=0; l<node->nlinks; l++) {
           struct ncclTopoLink* link = node->links+l;
           struct ncclTopoNode* remNode = link->remNode;
           if (remNode->paths[baseNode->type] == NULL) {
             NCCL_CHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
             for (int i=0; i<system->nodes[baseNode->type].count; i++) remNode->paths[baseNode->type][i].type = PATH_DIS;
           }
           struct ncclTopoLinkList* remPath;
           NCCL_CHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
           float bw = std::min(path->bw, link->bw);
 
           // allow routing through a GPU only as 1 hop
           if (node != baseNode && node->type == GPU &&
               (link->type != LINK_NVL || remNode->type != GPU || path->count > 1)) continue;
 
           if ((remPath->bw == 0 || remPath->count > path->count) && remPath->bw < bw) {
             // Find reverse link
             for (int l=0; l<remNode->nlinks; l++) {
               if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
                 remPath->list[0] = remNode->links+l;
                 break;
               }
             }
             if (remPath->list[0] == NULL) {
               // WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx",
               //      remNode->type, remNode->id, remNode->nlinks, node->type, node->id);
               return ncclInternalError;
             }
             // Copy the rest of the path
             for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
             remPath->count = path->count + 1;
             remPath->bw = bw;
             // Start with path type = link type. PATH and LINK types are supposed to match.
             // Don't consider LINK_NET as we only care about the NIC->GPU path.
             int type = link->type == LINK_NET ? LINK_LOC : link->type;
             // Differentiate between one and multiple PCI switches
             if (node->type == PCI && remNode->type == PCI) type = PATH_PXB;
             // Consider a path going through the CPU as PATH_PHB
             if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type = PATH_PHB;
             // Set 1 hop NVLink as NVB
             if (node->type == GPU && path->type == PATH_NVL && type == PATH_NVL && remPath->count > 1) type = PATH_NVB;
             remPath->type = std::max(path->type, type);
             // Add to the list for the next iteration if not already in the list
             int i;
             for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
             if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
           }
         }
       }
       memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index) {
     *index = -1;
     for (int i=0; i<system->nodes[GPU].count; i++) {
       if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
         *index = i;
         return ncclSuccess;
       }
     }
     return ncclInternalError;
   }
 
   ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
     // Shift all links to have upLink as last link
     if (upNode) {
       int l=0;
       while (node->links[l].remNode != upNode) l++;
       struct ncclTopoLink upLink;
       memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
       while (node->links[l+1].remNode) {
         memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
         l++;
       }
       memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
     }
     // Recursively sort the PCI tree
     for (int l=0; l<node->nlinks; l++) {
       struct ncclTopoLink* link = node->links+l;
       if (link->type == LINK_PCI && link->remNode != upNode) NCCL_CHECK(ncclTopoSort(link->remNode, node));
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
     for (int n=0; n<system->nodes[CPU].count; n++) NCCL_CHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
     return ncclSuccess;
   }
 
   int construct_xml_stru(struct ncclXml **xml, const std::vector<int> &local_rank_vec) {
     nvmlInit();
     assert(local_rank_vec.size() < MAX_SUBS);
     xmlAlloc(xml, NCCL_GRAPH_XML_MAX_NODES);
     struct ncclXmlNode* top;
     xmlAddNode(*xml, NULL, "system", &top);
     for (int idx = 0; idx < local_rank_vec.size(); ++idx) {
       char busIdStr[] = "00000000:00:00.0";
       cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), local_rank_vec[idx]);
       struct ncclXmlNode *node;
       ncclTopoFillGpu(*xml, busIdStr, &node);
       xmlSetAttrInt(node, "rank", idx);
       xmlSetAttrInt(node, "gdr", 1);
     }
     ibv_fork_init();
     int num_of_device;
     struct ibv_device **dev_list;
     struct ibv_device *ib_dev = nullptr;
     dev_list = ibv_get_device_list(&num_of_device);
     int nic_idx = 0;
     for (; ib_dev = *dev_list; ++dev_list) {
       struct ibv_context *context = ibv_open_device(ib_dev);
       struct ibv_device_attr devAttr;
       memset(&devAttr, 0, sizeof(devAttr));
       ibv_query_device(context, &devAttr);
       int port_cnt = devAttr.phys_port_cnt;
       for (int port_idx = 1; port_idx <= port_cnt; ++port_idx) {
         struct ibv_port_attr portAttr;
         ibv_query_port(context, port_idx, &portAttr);
         if (portAttr.state != IBV_PORT_ACTIVE) continue;
         if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
             && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
         char *pciPath = nullptr;
         ncclIbGetPciPath(ib_dev->name, &pciPath);
         struct ncclXmlNode* netNode;
         ncclTopoFillNet(*xml, pciPath, ib_dev->name, &netNode);
         xmlSetAttrInt(netNode, "keep", 1);
         xmlSetAttrInt(netNode, "dev", nic_idx++);
         xmlInitAttrInt(netNode, "speed", ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width));
         xmlInitAttrInt(netNode, "port", port_idx);
         xmlInitAttrFloat(netNode, "latency", 0.0f);
         xmlInitAttrUint64(netNode, "guid", devAttr.sys_image_guid);
         xmlInitAttrInt(netNode, "gdr", 1);
         break;
       }
       ibv_close_device(context);
     }
     // ibv_free_device_list(dev_list);
     return 0;
   }
 
   int construct_topo_stru(struct ncclXml* xml, struct ncclTopoSystem** topoSys) {
     *topoSys = (ncclTopoSystem *)calloc(1, sizeof(ncclTopoSystem));
     [[maybe_unused]] struct ncclTopoSystem *system = *topoSys;
     struct ncclXmlNode* topNode;
     xmlFindTag(xml, "system", &topNode);
     for (int s=0; s<topNode->nSubs; s++) {
       struct ncclXmlNode* node = topNode->subs[s];
       if (strcmp(node->name, "cpu") == 0) ncclTopoAddCpu(node, *topoSys);
     }
     ncclTopoConnectCpus(*topoSys);
     ncclTopoSortSystem(*topoSys);
     return 0;
   }
 
   int compute_paths(struct ncclTopoSystem *topoSys) {
     for (int idx = 0; idx < topoSys->nodes[GPU].count; ++idx) {
       ncclTopoSetPaths(topoSys->nodes[GPU].nodes + idx, topoSys);
     }
     for (int idx = 0; idx < topoSys->nodes[NET].count; ++idx) {
       ncclTopoSetPaths(topoSys->nodes[NET].nodes + idx, topoSys);
     }
     return 0;
   }
 
   int select_net(struct ncclTopoSystem* system, int gpuDev, const char **netStr) {
     int idx = 0;
     int localNets = 0;
     int localGpus = 0;
     int tempLocalGpus = 0;
     int gpuIdxInVec = -1;
     int localNetIndexes[NCCL_TOPO_MAX_NODES];
     int localGpuIndexes[NCCL_TOPO_MAX_NODES];
     int tempLocalGpuIndexes[NCCL_TOPO_MAX_NODES];
     int netPathType = 0;
     int gpuPathType = 0;
     ncclTopoRankToIndex(system, gpuDev, &idx);
     ncclTopoGetLocal(system, GPU, idx, NET, localNetIndexes, &localNets, &netPathType);
     for (int idx = 0; idx < localNets; ++idx) {
       ncclTopoGetLocal(system, NET, localNetIndexes[idx], GPU, tempLocalGpuIndexes, &tempLocalGpus, &gpuPathType);
       for (int new_idx = 0; new_idx < tempLocalGpus; ++new_idx) {
         for (int curr_idx = 0; curr_idx < localGpus; ++curr_idx) {
           if (localGpuIndexes[curr_idx] == tempLocalGpuIndexes[new_idx]) continue;
         }
         localGpuIndexes[localGpus++] = tempLocalGpuIndexes[new_idx];
       }
     }
     for (int idx = 0; idx < localGpus; ++idx) {
       if (gpuDev == localGpuIndexes[idx]) gpuIdxInVec = idx;
     }
     *netStr = system->nodes[NET].nodes[localNetIndexes[gpuIdxInVec % localNets]].net.name;
     return 0;
   }
   } //namespace
 
 inline int get_nic(const std::vector<int> &gpu_idx_vec, int gpu_idx, const char **netName) {
   struct ncclXml *xml;
   construct_xml_stru(&xml, gpu_idx_vec);
   struct ncclTopoSystem *topoSys;
   construct_topo_stru(xml, &topoSys);
   compute_paths(topoSys);
   select_net(topoSys, gpu_idx, netName);
   return 0;
 }
 } //namespace hybrid_ep
 #endif //HYBRID_EP_BUILD_MULTINODE_ENABLE
 