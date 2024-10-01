# Parameters:

| Parameter Name   | Type   | Description                                                    | Default Value |
|------------------|--------|----------------------------------------------------------------|---------------|
| vecDim           | Int64  | Dimension of the vector.                                       | 521           |
| maxConnection    | Int64  | Max connections of each node.                                  | 16            |
| metricType       | String | Distance metric type.                                          | L2            |
| efConstruction   | Int64  | Size of the dynamic candidate.                                 | 200           |
| optMode          | Int64  | Enable optimization or not.                                    | 0             |
| samplingStep     | Int64  | Optimization parameter,                                        | 64            |
| adsEpsilon       | Double | Optimization parameter,                                        | 1.0           |
| enableCoroutine  | Int64  | Enable conroutine or not.                                      | 1             |
| threadCount      | Int64  | Number of the threads.                                         | 10            |
| readThreadCount  | Int64  | Number of the read threads, must be less than readThreadCount. | 0             |
| cutOffTimeInSec  | Int64  | Setting time to cut off execution after given seconds.         | 1             |
| intialRows       | Int64  | The rows of initially loaded tensors.                          | 0             |
| waitPendingWrite | Int64  | Wether or not wait for pending writes before start a query.    | 0             |
| dataLoaderTag    | String | The name tag of data loader class.                             | random        |
| indexTag         | String | The name tag of index class                                    | hnsw          |




