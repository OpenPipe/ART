// CPU-built C ABI. No CUDA operation occurs before lt_new() in a granted worker.
#include <cublasLt.h>
#include <cstdint>
#include <cstring>
#include <new>
#include <initializer_list>
struct Context {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t a{},b{},c{};
  ~Context() {
    if(a) cublasLtMatrixLayoutDestroy(a);
    if(b) cublasLtMatrixLayoutDestroy(b);
    if(c) cublasLtMatrixLayoutDestroy(c);
    if(desc) cublasLtMatmulDescDestroy(desc);
    if(handle) cublasLtDestroy(handle);
  }
};
#define CHECK(x) do { int status=(x); if(status) return status; } while(0)
extern "C" {
int lt_abi_bytes() { return sizeof(cublasLtMatmulAlgo_t); }
size_t lt_version() { return cublasLtGetVersion(); }
int lt_new(int m,int n,int k,int bf16,int transb,int lda,int ldb,void** out) {
  auto* p=new(std::nothrow) Context; if(!p) return -1;
  *out=p; // Caller destroys even a partially initialized context.
  CHECK(cublasLtCreate(&p->handle));
  auto compute=bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC;
  CHECK(cublasLtMatmulDescCreate(&p->desc,compute,CUDA_R_32F));
  cublasOperation_t op=transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  CHECK(cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_TRANSB,&op,sizeof(op)));
  cublasLtEpilogue_t ep=CUBLASLT_EPILOGUE_DEFAULT;
  CHECK(cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));
  cudaDataType_t type=bf16 ? CUDA_R_16BF : CUDA_R_32F;
  CHECK(cublasLtMatrixLayoutCreate(&p->a,type,m,k,lda));
  CHECK(cublasLtMatrixLayoutCreate(&p->b,type,transb?n:k,transb?k:n,ldb));
  CHECK(cublasLtMatrixLayoutCreate(&p->c,type,m,n,n));
  cublasLtOrder_t order=CUBLASLT_ORDER_ROW;
  for(auto d:{p->a,p->b,p->c})
    CHECK(cublasLtMatrixLayoutSetAttribute(d,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)));
  return 0;
}
void lt_delete(void* ctx) { delete static_cast<Context*>(ctx); }
int lt_discover(void* ctx,size_t workspace,uint8_t* bytes,int* count) {
  auto* p=static_cast<Context*>(ctx);
  cublasLtMatmulPreference_t pref;
  CHECK(cublasLtMatmulPreferenceCreate(&pref));
  int s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
  if(s) {cublasLtMatmulPreferenceDestroy(pref);return s;}
  cublasLtMatmulHeuristicResult_t result[4]{};
  s=cublasLtMatmulAlgoGetHeuristic(p->handle,p->desc,p->a,p->b,p->c,p->c,pref,4,result,count);
  cublasLtMatmulPreferenceDestroy(pref);
  if(s) return s;
  for(int i=0;i<*count;i++) std::memcpy(bytes+i*sizeof(result[i].algo),&result[i].algo,sizeof(result[i].algo));
  return 0;
}
int lt_nosplit(uint8_t* bytes,int* accepted) {
  auto* algo=reinterpret_cast<cublasLtMatmulAlgo_t*>(bytes);
  uint32_t split=1,scheme=CUBLASLT_REDUCTION_SCHEME_NONE;
  CHECK(cublasLtMatmulAlgoConfigSetAttribute(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)));
  CHECK(cublasLtMatmulAlgoConfigSetAttribute(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme)));
  size_t written=0;
  CHECK(cublasLtMatmulAlgoConfigGetAttribute(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,accepted,sizeof(*accepted),&written));
  return *accepted==1 ? 0 : -2;
}
int lt_config(uint8_t* bytes,int attribute,int* value) {
  size_t written=0;
  return cublasLtMatmulAlgoConfigGetAttribute(reinterpret_cast<cublasLtMatmulAlgo_t*>(bytes),
    static_cast<cublasLtMatmulAlgoConfigAttributes_t>(attribute),value,sizeof(*value),&written);
}
int lt_check(void* ctx,uint8_t* bytes,size_t* workspace) {
  auto* p=static_cast<Context*>(ctx);cublasLtMatmulHeuristicResult_t result{};
  CHECK(cublasLtMatmulAlgoCheck(p->handle,p->desc,p->a,p->b,p->c,p->c,
    reinterpret_cast<cublasLtMatmulAlgo_t*>(bytes),&result));
  if(result.state) return result.state;
  *workspace=result.workspaceSize;return 0;
}
int lt_alignment(uint8_t* bytes,int operand,int* value) {
  size_t written=0;
  return cublasLtMatmulAlgoCapGetAttribute(reinterpret_cast<cublasLtMatmulAlgo_t*>(bytes),
    static_cast<cublasLtMatmulAlgoCapAttributes_t>(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES+operand),
    value,sizeof(*value),&written);
}
int lt_run(void* ctx,uint8_t* bytes,void* a,void* b,void* c,void* workspace,size_t size,void* stream) {
  auto* p=static_cast<Context*>(ctx);float alpha=1,beta=0;
  // Always pass the caller's admitted algorithm. There is no heuristic fallback.
  return cublasLtMatmul(p->handle,p->desc,&alpha,a,p->a,b,p->b,&beta,c,p->c,c,p->c,
    reinterpret_cast<cublasLtMatmulAlgo_t*>(bytes),workspace,size,reinterpret_cast<cudaStream_t>(stream));
}
}
