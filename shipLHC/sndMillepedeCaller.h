#ifndef SHIPLHC_SNDMILLEPEDECALLER_H_
#define SHIPLHC_SNDMILLEPEDECALLER_H_

#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
//includes for GBL fitter from genfit
#include <vector>
#include <stdexcept>
#include "GblPoint.h"
#include "GblTrajectory.h"
#include "MilleBinary.h"
#include "Track.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include <map>
#include <unordered_map>
#include "TVector3.h"
#include "TDecompLU.h"
#include "TRotation.h"
#include "TMath.h"
#include <cstdint>
//#include "MufluxSpectrometer.h"
#include "sndScifiHit.h"
#include <iostream>
#include "GBLseedtrack.h"
#include "Scifi.h"
#include "TH1D.h"
#include <list>

//includes for MC testing
#include <random>
#include <fstream>



struct VectorTriple
{
	std::unordered_map<std::string,TVector3> mat_center;
	std::unordered_map<std::string,TVector3> mat_top;
	std::unordered_map<std::string,TVector3> mat_bot;
};

struct VectorQuadInfo
{
	std::vector<TVector3> mat_center;
	std::vector<TVector3> mat_bot;
	std::vector<TVector3> mat_top;
	std::vector<int> det_id;

};





class sndMillepedeCaller//: public TObject
{

public:

	sndMillepedeCaller(const string out_file_name);
	virtual ~sndMillepedeCaller();

	gbl::GblTrajectory perform_GBL_refit(const GBL_seed_track& track, double sigma_spatial, map<int, double>* pede_corrections = nullptr, const char* spillname = nullptr);

	// Accessor functions for residuals, meas_errors, res_errors, down_weights
	const std::vector<double>& getResiduals() const { return residuals_; }
	const std::vector<double>& getMeasurementErrors() const { return meas_errors_; }
	const std::vector<double>& getResidualErrors() const { return res_errors_; }
	const std::vector<double>& getDownWeights() const { return down_weights_; }

	void clearStoredData() 
	{
        	residuals_.clear();
        	meas_errors_.clear();
        	res_errors_.clear();
        	down_weights_.clear();
    }

    std::string formatMatKey(int det_id);

    std::unordered_map<std::string, TVector3> m_nominal_mat_centerpos;
    std::unordered_map<std::string, TVector3> m_nominal_mat_toppos;
    std::unordered_map<std::string, TVector3> m_nominal_mat_botpos;

    std::unordered_map<std::string, std::vector<int>> m_mats;
    std::unordered_map<int, std::string> m_channel_id_to_mat;

private:

	gbl::MilleBinary* m_gbl_mille_binary;
	//std::unordered_map<int, std::string> m_channel_id_to_mat;
	//std::unordered_map<std::string, std::vector<int>> m_mats; //detector IDs making up a mat
	//std::unordered_map<std::string, TVector3> m_nominal_mat_centerpos; // nominal geometric center of a mat
	TFile* m_output_file;
	TTree* m_output_tree;
	Scifi m_scifi;

	std::vector<double> residuals_;
    std::vector<double> meas_errors_;
    std::vector<double> res_errors_;
    std::vector<double> down_weights_;

	

	// Helpers for projection and residuals

	TVector3 calc_shortest_distance(const TVector3& fiber_top, const TVector3& fiber_bot, const TVector3& track_pos, const TVector3& track_mom,  TVector3* PCA_on_fiber = nullptr, TVector3* PCA_on_track = nullptr) const;
	TRotation calc_rotation_of_vector(const TVector3& v) const;
	TMatrixD TRot_to_TMatrixD(const TRotation& rot) const;
	TMatrixD calc_projection_matrix(const TMatrixD& fit_system_base_vectors, const TMatrixD& rotation_global_to_measurement) const;
	void get_mat_pos();
	TMatrixD* calc_jacobian(const TVector3& PCA_1, const TVector3& PCA_2) const;
	


	// Helpers for GBL

	void add_measurement_info(gbl::GblPoint& point, const TVector3& closest_approach, const double measurement, const double sigma_spatial) const;
	std::vector<TVector3> linear_model_wo_scatter(const genfit::Track& track) const;
	void print_model_parameters(const std::vector<TVector3>& model) const;
	std::vector<gbl::GblPoint> list_hits(const GBL_seed_track* track, double sigma_spatial, std::map<int,double>* pede_corrections = nullptr, TTree* tree = nullptr);


	// Helpers for Pede 

	TMatrixD* calc_global_parameters(const TVector3& measurement_prediction, const TVector3& closest_approach, const vector<TVector3>& linear_model, const TVector3& fiber_bot_to_top) const;
	std::vector<int> calc_labels(const int det_id) const;
	


	// Debugging

	TTree* create_output_tree();
	bool check_ordered_by_arclen(const genfit::Track& track) const;
	void print_seed_hits(const genfit::Track& track) const;
	void print_fitted_residuals(gbl::GblTrajectory& trajectory) const;
	void print_fitted_track(gbl::GblTrajectory& trajectory) const;


};

#endif /* SHIPLHC_SNDMILLEPEDECALLER_H_ */
