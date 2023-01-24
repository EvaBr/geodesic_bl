#include "image.h"
#include <queue>
#include <vector>
#include <math.h>
typedef unsigned char uint8;
typedef image<uint8> imagetype;
typedef void (imagetype::*NeighborhoodFunction)(int i, vector<int> &, vector<int> &);
#include "Distances.hpp"
using namespace std;

class Qelem
{
public:
  double maxvalue;
  double minvalue;
  long index;

  Qelem(long i, double min, double max)
  {
    maxvalue = max;
    minvalue = min;
    index = i;
  }
};

class ElemComparison
{
public:
  bool operator()(const Qelem &lhs, const Qelem &rhs) const
  {
    if (lhs.maxvalue == rhs.maxvalue)
    {
      return lhs.minvalue < rhs.minvalue;
    }
    return lhs.maxvalue > rhs.maxvalue;
  }
};

typedef priority_queue<Qelem, vector<Qelem>, ElemComparison> Pq;

void MBD_exact(imagetype seeds, imagetype cost, double *dt, NeighborhoodFunction getN)
{

  int nelem = seeds.getNelem();
  bool *inQ = new bool[nelem];
  int *dtmin = new int[nelem];
  int *dtmax = new int[nelem];
  Pq Q;

  for (int i = 0; i < nelem; i++)
  {
    dt[i] = 256; // INF
    dtmin[i] = -1;
    dtmax[i] = 256;
    inQ[i] = false;
    if (seeds[i] > 0)
    {
      dt[i] = 0;
      dtmin[i] = cost[i];
      dtmax[i] = cost[i];
      Q.push(Qelem(i, cost[i], cost[i]));
      inQ[i] = true;
    }
  }

  int ni;
  double minVal, maxVal;
  vector<int> neighbors;
  vector<int> toadd;

  while (!Q.empty())
  {
    Qelem e = Q.top();
    Q.pop();

    if (inQ[e.index])
    {
      inQ[e.index] = false;
      // neighbors = seeds.get_8_neighbors(e.index);
      // seeds.get_26_neighbors_py(e.index, neighbors, toadd);
      (seeds.*getN)(e.index, neighbors, toadd);

      for (int i = 0; i < (int)neighbors.size(); i++)
      {
        ni = neighbors[i];

        minVal = min(dtmin[e.index], int(cost[ni]));
        maxVal = max(dtmax[e.index], int(cost[ni]));

        if (minVal > dtmin[ni])
        {
          dtmin[ni] = minVal;
          dtmax[ni] = maxVal;

          if (maxVal - minVal < dt[ni])
          {
            dt[ni] = maxVal - minVal;
          }

          Q.push(Qelem(ni, minVal, maxVal));
          inQ[ni] = true;
        }
      }
    }
  }
  delete[] inQ;
  delete[] dtmin;
  delete[] dtmax;
}

void GeodesicDT(imagetype seeds, imagetype cost, double *dt, NeighborhoodFunction getN)
{

  int nelem = seeds.getNelem();
  bool *inQ = new bool[nelem];
  Pq Q;

  for (int i = 0; i < nelem; i++)
  {
    dt[i] = 100000; // INF
    inQ[i] = false;

    if (seeds[i] > 0)
    {
      dt[i] = 0;
      Q.push(Qelem(i, 0, 0));
      inQ[i] = true;
    }
  }

  int ni;
  double fval;
  vector<int> neighbors;
  vector<int> toadd;

  while (!Q.empty())
  {
    Qelem e = Q.top();
    Q.pop();

    if (inQ[e.index])
    {
      inQ[e.index] = false;
      // seeds.get_8_neighbors_py(e.index, neighbors, toadd);
      // seeds.get_26_neighbors_py(e.index, neighbors, toadd); /* TODO: use _26 for 3D, but _8 for 2D!*/
      (seeds.*getN)(e.index, neighbors, toadd);

      for (int i = 0; i < (int)neighbors.size(); i++)
      {
        ni = neighbors[i];
        fval = dt[e.index] + sqrt(pow(double(cost[ni]) - double(cost[e.index]), 2) + double(toadd[i])); // abs(double(cost[ni])-double(cost[e.index])) + 1;
        // fval = dt[e.index] + 0.5 * abs(double(cost[ni]) - double(cost[e.index])) + 0.5 * sqrt(double(toadd[i]));
        if (fval < dt[ni])
        {
          dt[ni] = fval;
          Q.push(Qelem(ni, 0, fval));
          inQ[ni] = true;
        }
      }
    }
  }
  delete[] inQ;
}

void MBD(uint8 *seed, uint8 *cost, int height, int width, int channels, double *dt, int neighborhood)
{
  /* height = size of 1st dim */
  /* width =  size of 2nd dim */
  /* channels = size of 3rd dim*/

  // Setup input

  /*first arg for imagetype creation needs to be a pointer to uint8 data. How to get that??*/
  imagetype seeds = imagetype(seed, height, width, channels); /*border*/
  imagetype costs = imagetype(cost, height, width, channels); /*input image*/

  // Setup output
  // double* dt in output. Computation done in place

  NeighborhoodFunction Fn;
  // EXAMPLE of accessing member function:
  // define: void (Testpm::*pmfn)() = &Testpm::m_func1;
  // call: (ATestpmInstance.*pmfn)();
  if (neighborhood == 26)
  { // ful connectivity, 3d
    Fn = &imagetype::get_26_neighbors_py;
  }
  else if (neighborhood == 8)
  { // ful connectivity, 2d
    Fn = &imagetype::get_8_neighbors_py;
  }
  else if (neighborhood == 4)
  { // only direct connectivity, 2d (also semi)
    Fn = &imagetype::get_4_neighbors_py;
  }
  else if (neighborhood == 6)
  { // only direct connectivity, 2d
    Fn = &imagetype::get_6_neighbors_py;
  }
  else
  {
    // we're in 3d, 'semi' connectivity
    Fn = &imagetype::get_18_neighbors_py;
  }

  // Calculate DT
  MBD_exact(seeds, costs, dt, Fn);
}

void GEO(uint8 *seed, uint8 *cost, int height, int width, int channels, double *dt, int neighborhood)
{
  /* height = size of 1st dim */
  /* width =  size of 2nd dim */
  /* channels = size of 3rd dim*/

  // Setup input

  /*first arg for imagetype creation needs to be a pointer to uint8 data. How to get that??*/
  imagetype seeds = imagetype(seed, height, width, channels); /*border*/
  imagetype costs = imagetype(cost, height, width, channels); /*input image*/

  NeighborhoodFunction Fn;
  // EXAMPLE of accessing member function:
  // define: void (Testpm::*pmfn)() = &Testpm::m_func1;
  // call: (ATestpmInstance.*pmfn)();
  if (neighborhood == 26)
  { // ful connectivity, 3d
    Fn = &imagetype::get_26_neighbors_py;
  }
  else if (neighborhood == 8)
  { // ful connectivity, 2d
    Fn = &imagetype::get_8_neighbors_py;
  }
  else if (neighborhood == 4)
  { // only direct connectivity, 2d (also semi)
    Fn = &imagetype::get_4_neighbors_py;
  }
  else if (neighborhood == 6)
  { // only direct connectivity, 2d
    Fn = &imagetype::get_6_neighbors_py;
  }
  else
  {
    // we're in 3d, 'semi' connectivity
    Fn = &imagetype::get_18_neighbors_py;
  }
  // Calculate DT
  GeodesicDT(seeds, costs, dt, Fn);
}
