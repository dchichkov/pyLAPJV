/*
  pyLAPJV by Harold Cooper (hbc@mit.edu)
  pyLAPJV.cpp 2004-08-20
*/

#include "Python.h"
#include "numpy/arrayobject.h"
#include "lap.h"

static PyObject *
LAPJV_lap(PyObject *self, PyObject *args)
//lap(costs)
{
  PyObject *ocosts;
  PyArrayObject *costs;
  int n;
  npy_intp n2;
  long *rowsol;
  long *colsol;
  cost lapcost,*buf,**ccosts,*u,*v;
  npy_intp *strides;
  PyObject * rowo;
  PyObject * colo;

  if (!PyArg_ParseTuple(args, "O", &ocosts))
    return NULL;
  costs = (PyArrayObject*)PyArray_FromAny(
                                          ocosts,PyArray_DescrFromType(COST_TYPE_NPY),2,2,
                                          NPY_CONTIGUOUS|NPY_ALIGNED|NPY_FORCECAST,0
                                          );
  if (costs->nd!=2)
    {
      PyErr_SetString(PyExc_ValueError,"lap() requires a 2-D matrix");
      goto error;
    }
  n = costs->dimensions[0];
  n2 = costs->dimensions[0];
  if (costs->dimensions[1]!=n)
    {
      PyErr_SetString(PyExc_ValueError,"lap() requires a square matrix");
      goto error;
    }

  //get inputted matrix as a 1-D C array:
  //buf = (cost *)NA_OFFSETDATA(costs);
  buf = (cost*)PyArray_DATA(costs);

  //copy inputted matrix into a 2-dimensional C array:
  strides = PyArray_STRIDES(costs);
  assert(strides[1] == sizeof(cost));
  ccosts = (cost **)malloc(sizeof(cost *)*n);
  if(!ccosts)
    {
      PyErr_NoMemory();
      free(ccosts);
      goto error;
    }
  for(int i=0;i<n;i++)
    ccosts[i] = buf+i*(strides[0]/sizeof(cost));

  rowo = PyArray_SimpleNew(1, &n2, NPY_LONG);
  colo = PyArray_SimpleNew(1, &n2, NPY_LONG);
  rowsol = (long *) PyArray_DATA(rowo);
  colsol = (long *) PyArray_DATA(colo);
  u = (cost *)malloc(sizeof(cost)*n);
  v = (cost *)malloc(sizeof(cost)*n);
  if(!(rowsol&&colsol&&u&&v))
    {
      PyErr_NoMemory();
      free(ccosts);
      goto error;
    }

  //run LAPJV!:
  lapcost = lap(n,ccosts,rowsol,colsol,u,v);

  //NA_InputArray() incremented costs, but now we're done with it, so let it get GC'ed:
  Py_XDECREF(costs);

  free(ccosts);
  return Py_BuildValue("(NN)",
                       rowo, colo
                       );
 error:
  Py_XDECREF(costs);
  return NULL;
}

static PyMethodDef LAPJVMethods[] = {
  {"lap",  LAPJV_lap, METH_VARARGS,
   "Solves the linear assignment problem using the Jonker-Volgenant\nalgorithm.\n\nlap() takes a single argument - a square cost matrix - and returns a\ntuple of the form\n(row_assigns,col_assigns).\n\nThe average user probably just wants the second or third of these\nelements, so a call like: \'assigns=lap(costs)[1]' would be the\nmost common use."},
  {NULL, NULL, 0, NULL}        /* Sentinel (terminates structure) */
};

PyMODINIT_FUNC
initLAPJV(void)
{
  (void) Py_InitModule("LAPJV", LAPJVMethods);
  import_array();
}

int
main(int argc, char *argv[])
{
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(argv[0]);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  initLAPJV();

  return 0;
}
