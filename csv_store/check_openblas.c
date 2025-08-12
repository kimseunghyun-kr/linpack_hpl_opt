#include <stdio.h>
#ifdef __cplusplus
extern "C"
#endif
const char* openblas_get_config(void);
int main(){ if(openblas_get_config) printf("%s\n",openblas_get_config()); return 0; }
