/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See NCCL_LICENSE.txt for license information
 ************************************************************************/

 #pragma once

 #ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
 #include <arpa/inet.h>
 #include <fcntl.h>
 #include <infiniband/verbs.h>
 #include <sys/socket.h>
 #include "utils.cuh"
 
 namespace hybrid_ep {
   namespace {
   static const int IB_GID_INDEX = -1;
   static const int IB_ROUTABLE_FLID_GID_INDEX = 1;
   static const int IB_ROCE_VERSION_NUM = 2;
   static const sa_family_t DEFAULT_FAMILY = AF_INET;
   static const char NCCL_IB_ADDR_RANGE[] = { 0 };
 
   int ncclIbExtractFlid(union ibv_gid *gid) {
     return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
   }
 
   static void* envIbAddrRange(sa_family_t af, int* mask) {
     *mask = 0;
     static struct in_addr addr;
     static struct in6_addr addr6;
     void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;
     const char* env = NCCL_IB_ADDR_RANGE;
     if (NULL == env || strlen(env) == 0) {
       return NULL;
     }
     // INFO(NCCL_ENV, "NCCL_IB_ADDR_RANGE set by environment to %s", env);
     char addrString[128] = { 0 };
     snprintf(addrString, 128, "%s", env);
     char *addrStrPtr = addrString;
     char *maskStrPtr = strstr(addrString, "/");
     if (NULL == maskStrPtr) {
       return NULL;
     }
     *(maskStrPtr++) = '\0';
     if (inet_pton(af, addrStrPtr, ret) == 0) {
       // INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
       return NULL;
     }
     *mask = (int)strtol(maskStrPtr, NULL, 10);
     if (af == AF_INET && *mask > 32) {
       // INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
       *mask = 0;
       ret = NULL;
     } else if (af == AF_INET6 && *mask > 128) {
       // INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
       *mask = 0;
       ret = NULL;
     }
     return ret;
   }
 
   sa_family_t getGidAddrFamily(union ibv_gid* gid) {
     const struct in6_addr *a = (struct in6_addr *)gid->raw;
     bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
     bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
     return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
   }
 
   bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
     struct in_addr *base = NULL;
     struct in6_addr *base6 = NULL;
     struct in6_addr *addr6 = NULL;;
     if (af == AF_INET) {
       base = (struct in_addr *)prefix;
     } else {
       base6 = (struct in6_addr *)prefix;
     }
     addr6 = (struct in6_addr *)gid->raw;
 #define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))
     int i = 0;
     while (prefixlen > 0 && i < 4) {
       if (af == AF_INET) {
         int mask = NETMASK(prefixlen);
         if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
           break;
         }
         prefixlen = 0;
         break;
       } else {
         if (prefixlen >= 32) {
           if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
             break;
           }
           prefixlen -= 32;
           ++i;
         } else {
           int mask = NETMASK(prefixlen);
           if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
             break;
           }
           prefixlen = 0;
         }
       }
     }
     return (prefixlen == 0) ? true : false;
   }
 
   bool configuredGid(union ibv_gid* gid) {
     const struct in6_addr *a = (struct in6_addr *)gid->raw;
     int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
     if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
       return false;
     }
     return true;
   }
 
   bool linkLocalGid(union ibv_gid* gid) {
     const struct in6_addr *a = (struct in6_addr *)gid->raw;
     if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
       return true;
     }
     return false;
   }
 
   bool validGid(union ibv_gid* gid) {
     return (configuredGid(gid) && !linkLocalGid(gid));
   }
 
   ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
     char gidRoceVerStr[16] = { 0 };
     char roceTypePath[PATH_MAX] = { 0 };
     snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);
     int fd = open(roceTypePath, O_RDONLY);
     if (fd == -1) {
       // WARN("NET/IB: open failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
       return ncclSystemError;
     }
     int ret = read(fd, gidRoceVerStr, 15);
     close(fd);
     if (ret == -1) {
       // In containerized environments, read could return EINVAL if the GID index is not mapped to the
       // container sysfs. In this case return ncclSuccess and let the caller move to next GID index.
       if (errno == EINVAL) return ncclSuccess;
       // WARN("NET/IB: read failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
       return ncclSystemError;
     }
     if (strlen(gidRoceVerStr)) {
       if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
         *version = 1;
       } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
         *version = 2;
       }
     }
     return ncclSuccess;
   }
 
   ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex) {
     union ibv_gid gid, gidCandidate;
     CALL_CHECK(ibv_query_gid(context, portNum, *gidIndex, &gid));
     CALL_CHECK(ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));
     sa_family_t usrFam = af;
     sa_family_t gidFam = getGidAddrFamily(&gid);
     sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
     bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);
     if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
       *gidIndex = gidIndexCandidate;
     } else {
       if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
         return ncclSuccess;
       }
       int usrRoceVer = roceVer;
       int gidRoceVerNum, gidRoceVerNumCandidate = -1;
       const char* deviceName = ibv_get_device_name(context->device);
       NCCL_CHECK(ncclIbRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
       NCCL_CHECK(ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
       if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
         *gidIndex = gidIndexCandidate;
       }
     }
 
     return ncclSuccess;
   }
 
   }
 
 static ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex) {
   int gidTblLen = portAttr->gid_tbl_len;
   //for IB, choose GID Index that will have routable FLID if present
   if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
     union ibv_gid gid;
     int routableGidIndex = IB_ROUTABLE_FLID_GID_INDEX;
     if (routableGidIndex < gidTblLen) {
       CALL_CHECK(ibv_query_gid(context, portNum, routableGidIndex, &gid));
       if (ncclIbExtractFlid(&gid) != 0) {
         *gidIndex = routableGidIndex;
         return ncclSuccess;
       }
     }
     *gidIndex = 0;
     return ncclSuccess;
   }
   //for ROCE
   *gidIndex = IB_GID_INDEX;
   if (*gidIndex >= 0) {
     return ncclSuccess;
   }
   sa_family_t userAddrFamily = DEFAULT_FAMILY;
   int userRoceVersion = IB_ROCE_VERSION_NUM;
   int prefixlen;
   void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);
   *gidIndex = 0;
   for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
     NCCL_CHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext, gidIndex));
   }
 
   return ncclSuccess;
 }
 } //namespace hybrid_ep
 #endif //HYBRID_EP_BUILD_MULTINODE_ENABLE
 