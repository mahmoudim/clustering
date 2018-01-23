/*
 * SpectralClustering.cpp
 *
 *  Created on: 04-Mar-2009
 *      Author: sbutler
 */

#include "SpectralClustering.h"
#include "ClusterRotate.h"
#include "Kmeans.h"
#include <MatOp/SparseSymMatProd.h>
#include<SymEigsSolver.h>

#include <Eigen/QR>

/**
* Performs eigenvector decomposition of an affinity matrix
*
* @param data 		the affinity matrix
* @param numDims	the number of dimensions to consider when clustering
*/
SpectralClustering::SpectralClustering(Eigen::SparseMatrix<double>& data, int numDims):
	mNumDims(numDims),
	mNumClusters(0)
{
	Eigen::SparseMatrix<double> Deg = Eigen::SparseMatrix<double>(data.rows(),data.cols());

	// calc normalised laplacian 
	for ( int i=0; i < data.cols(); i++) {
		Deg.insert(i,i)=1/(sqrt((data.row(i).sum())) );
	}
	Eigen::SparseMatrix<double> Lapla = Deg * data * Deg;

	//Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> *s= new Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>>(Lapla, numDims);
	//RedSVD::RedSymEigen<Eigen::SparseMatrix<double>> *s=new RedSVD::RedSymEigen<Eigen::SparseMatrix<double>>(Lapla, numDims);
	Spectra::SparseSymMatProd<double> op(Lapla);
	Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseSymMatProd<double>> solve(&op, numDims, 2 * numDims + 2);
	solve.init();
	int ss = solve.compute();
	if (solve.info() != Spectra::SUCCESSFUL)
		return;
	Eigen::VectorXd val = solve.eigenvalues();// s->eigenvalues();
	Eigen::MatrixXd vec = solve.eigenvectors();
	//printf(" ok ");
	//sort eigenvalues/vectors
	int n = numDims;
	for (int i = 0; i < n-1 ; ++i) {
		int k;
		val.segment(i, n - i).maxCoeff(&k);
		if (k > 0) {
			std::swap(val[i], val[k + i]);
			vec.col(i).swap(vec.col(k + i));
		}
	}

	//choose the number of eigenvectors to consider
	if (mNumDims < vec.cols()) {
		mEigenVectors = vec.block(0,0,vec.rows(),mNumDims);
	} else {
		mEigenVectors = vec;
	}
}

SpectralClustering::~SpectralClustering() {
}

/**
 * Cluster by rotating the eigenvectors and evaluating the quality
 */
std::vector<std::vector<int> > SpectralClustering::clusterRotate() {

	ClusterRotate* c = new ClusterRotate();
	std::vector<std::vector<int> > clusters = c->cluster(mEigenVectors);

	mNumClusters = clusters.size();

	return clusters;
}

/**
 * Cluster by kmeans
 *
 * @param numClusters	the number of clusters to assign
 */
std::vector<std::vector<int> > SpectralClustering::clusterKmeans(int numClusters) {
	mNumClusters = numClusters;
	return Kmeans::cluster(mEigenVectors, numClusters);
}
