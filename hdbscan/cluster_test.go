// Copyright 2020 Humility AI Incorporated, All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package hdbscan

import (
	"fmt"
	"sort"
	"testing"
)

var (
	data = [][]float64{
		// cluster-1 (0-7)
		{1, 2, 3},
		{1, 2, 4},
		{1, 2, 5},
		{1, 3, 4},
		{2, 3, 3},
		{2, 2, 4},
		{2, 2, 5},
		{2, 3, 4},
		// cluster-2 (8-15)
		{21, 15, 6},
		{22, 15, 5},
		{23, 15, 7},
		{24, 15, 8},
		{21, 15, 6},
		{22, 16, 5},
		{23, 17, 7},
		{24, 18, 8},
		// cluster-3 (16-23)
		{80, 85, 90},
		{89, 90, 91},
		{100, 100, 100}, // possible outlier
		{90, 90, 90},
		{81, 85, 90},
		{89, 91, 91},
		{100, 101, 100}, // possible outlier
		{90, 91, 90},
		// outlier
		{-2400, 2000, -30},
	}
	minimumClusterSize = 3
)

func TestMinimumSpanningTree(t *testing.T) {
	clustering, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	clustering.distanceFunc = EuclideanDistance
	clustering.minTree = true

	// graph
	fmt.Println(clustering.mutualReachabilityGraph())
}

func TestBuildDendrogram(t *testing.T) {
	clustering, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	clustering.distanceFunc = EuclideanDistance
	clustering.minTree = true

	// cluster-hierarchy
	dendrogram := clustering.buildDendrogram(clustering.mutualReachabilityGraph())

	for _, link := range dendrogram {
		t.Logf("Link %+v with points: %+v", link.id, link.points)
	}
}

func TestBuildClusters(t *testing.T) {
	clustering, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	clustering.distanceFunc = EuclideanDistance
	// clustering.minTree = true

	// cluster-hierarchy
	dendrogram := clustering.buildDendrogram(clustering.mutualReachabilityGraph())
	clustering.buildClusters(dendrogram)

	for _, cluster := range clustering.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusterScoring(t *testing.T) {
	clustering, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	clustering.distanceFunc = EuclideanDistance

	// cluster-hierarchy
	dendrogram := clustering.buildDendrogram(clustering.mutualReachabilityGraph())
	clustering.buildClusters(dendrogram)
	clustering.scoreClusters(VarianceScore)

	for _, cluster := range clustering.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with variance %+v and score %+v and points: %+v", cluster.id, cluster.variance, cluster.score, cluster.Points)
	}
}

func TestClustering(t *testing.T) {
	clustering, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}

	err = clustering.Run(EuclideanDistance, VarianceScore, true)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range clustering.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringNoTree(t *testing.T) {
	clustering, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}

	err = clustering.Run(EuclideanDistance, VarianceScore, false)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range clustering.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringVerbose(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Verbose()

	err = c.Run(EuclideanDistance, VarianceScore, false)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringSampling(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Verbose().Subsample(16)

	err = c.Run(EuclideanDistance, VarianceScore, true)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringSamplingAndAssign(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Subsample(16).OutlierDetection()

	err = c.Run(EuclideanDistance, VarianceScore, true)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	newClustering, err := c.Assign(data)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range newClustering.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringSamplingAndAssignAndOutlierClustering(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Subsample(16).NearestNeighbor().OutlierClustering()

	err = c.Run(EuclideanDistance, VarianceScore, true)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	newClustering, err := c.Assign(data)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range newClustering.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringOutliers(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}

	c = c.OutlierDetection().NearestNeighbor()

	err = c.Run(EuclideanDistance, VarianceScore, true)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with Points %+v and outliers: %+v", cluster.id, cluster.Points, cluster.Outliers)
	}
}

func TestClusteringVoronoi(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Verbose().Voronoi()

	err = c.Run(EuclideanDistance, VarianceScore, true)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}

func TestClusteringVoronoiParts(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Verbose().Voronoi()
	c.distanceFunc = EuclideanDistance
	c.minTree = true

	edges := c.mutualReachabilityGraph()
	t.Logf("%+v\n", edges)
	dendrogram := c.buildDendrogram(edges)
	for _, link := range dendrogram {
		t.Logf("Link %+v with points: %+v", link.id, link.points)
	}

	c.buildClusters(dendrogram)
	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}

	c.scoreClusters(VarianceScore)
	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with variance %+v and score %+v and points: %+v", cluster.id, cluster.variance, cluster.score, cluster.Points)
	}

	c.selectOptimalClustering(VarianceScore)
	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with variance %+v and score %+v and points: %+v", cluster.id, cluster.variance, cluster.score, cluster.Points)
	}

	c.clusterCentroids()
	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with variance %+v and score %+v and points: %+v and Centroid %+v", cluster.id, cluster.variance, cluster.score, cluster.Points, cluster.Centroid)
	}

	c.outliersAndVoronoi()
	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with variance %+v and score %+v and points: %+v and Centroid %+v", cluster.id, cluster.variance, cluster.score, cluster.Points, cluster.Centroid)
	}
}

func TestClusteringVoronoiNoTree(t *testing.T) {
	c, err := NewClustering(data, minimumClusterSize)
	if err != nil {
		t.Errorf("clustering creation error: %+v", err)
	}
	c = c.Verbose().Voronoi()

	err = c.Run(EuclideanDistance, VarianceScore, false)
	if err != nil {
		t.Errorf("clustering run error: %+v", err)
	}

	for _, cluster := range c.Clusters {
		sort.Ints(cluster.Points)
		t.Logf("Cluster %+v with points: %+v", cluster.id, cluster.Points)
	}
}
