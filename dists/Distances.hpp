#ifdef _MSC_VER
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C"
{
#endif
    typedef unsigned char uint8;

    EXPORT_SYMBOL void GEO(uint8 *seed, uint8 *cost, int height, int width, int channels, double *dt, int neighborhood);
    EXPORT_SYMBOL void MBD(uint8 *seed, uint8 *cost, int height, int width, int channels, double *dt, int neighborhood);

#ifdef __cplusplus
}
#endif