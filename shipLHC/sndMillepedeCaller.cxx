// Wrapper to the Mille class. 16.04.2024
#include "sndMillepedeCaller.h"
#include "Scifi.h"

using namespace std;

// Constructor 

sndMillepedeCaller::sndMillepedeCaller(const string out_file_name)
{

	std::cout << "In MillepedeCaller" << std::endl;
	m_gbl_mille_binary = new gbl::MilleBinary(out_file_name, true, 2000);
	
	stringstream root_output;
	root_output << out_file_name << ".root";
	m_output_file = new TFile(root_output.str().c_str(),"RECREATE");
	std::cout << "Output File created" << std::endl;
	
	m_output_tree = create_output_tree();
	std::cout << "Output Tree created" << std::endl;
	
	//Scifi m_scifi;
	//m_scifi.Initialize();

	/*  5 SciFi volumes, for each volume there are 3 Horizontal and 3 Vertical mats
		For each mat, there are 4 sipms with 128 channels per sipm
		volume > orientation > mat > sipm > channel

		Labeling: 'Volume''#''Horizontal/Vertical''Mat''#' 
		e.g: for SciFi Volume 1, Horizontal, Mat 3 --> V1HM2
	*/
        
        std::cout << "m_mats is being created " << std::endl;

	m_mats["V1HM0"] = {};	
	m_mats["V1HM1"] = {};
	m_mats["V1HM2"] = {};
	m_mats["V1VM0"] = {};
	m_mats["V1VM1"] = {};
	m_mats["V1VM2"] = {};
	m_mats["V2HM0"] = {};
	m_mats["V2HM1"] = {};
	m_mats["V2HM2"] = {};
	m_mats["V2VM0"] = {};
	m_mats["V2VM1"] = {};
	m_mats["V2VM2"] = {};
	m_mats["V3HM0"] = {};
	m_mats["V3HM1"] = {};
	m_mats["V3HM2"] = {};
	m_mats["V3VM0"] = {};
	m_mats["V3VM1"] = {};
	m_mats["V3VM2"] = {};
	m_mats["V4HM0"] = {};
	m_mats["V4HM1"] = {};
	m_mats["V4HM2"] = {};
	m_mats["V4VM0"] = {};
	m_mats["V4VM1"] = {};
	m_mats["V4VM2"] = {};
	m_mats["V5HM0"] = {};
	m_mats["V5HM1"] = {};
	m_mats["V5HM2"] = {};
	m_mats["V5VM0"] = {};
	m_mats["V5VM1"] = {};
	m_mats["V5VM2"] = {};


        std::cout << "m_mats creation is finished " << std::endl;

	for(auto vec:m_mats)
	{
		vec.second.reserve(512); // 128 x 4 = 512 channels per mat
	}

        std::cout << "reserved spots for each channel " << std::endl;       

	// Generate list of IDs
	
	for (char volume = 1; volume < 6; volume++)
	{
		for (char orientation = 0; orientation < 2; orientation++)
		{
			for (char mat = 0; mat < 3; mat++)
			{
				for (char sipm = 0; sipm < 4; sipm++)
				{
					for (int channel = 0; channel < 128; channel++)
					{

						stringstream mat_key;
						mat_key << "V" << (int)volume;

						if(orientation == 0)
						{
							mat_key << "HM";
						}
						else
						{
							mat_key << "VM";
						}

						string key = mat_key.str();

						int id = (volume * 1000000) + (orientation * 100000) + (mat * 10000) + (sipm * 1000) + (int)channel;
						
                                                
                                                std::cout << "Volume: " << (volume * 1000000) << ", "
                                                          << "Orientation: " << (orientation * 100000) << ", "
                                                          << "Mat: " << (mat * 10000) << ", "
                                                          << "SIPM: " << (sipm * 1000) << ", "
                                                          << "Channel: " << channel << std::endl;

                                                std::cout << "ID: " << id << std::endl;

						m_mats[key].push_back(id);
						m_channel_id_to_mat[id] = key;

					}
				}
			}
		}
	}

        std::cout << "channel ids are generated successfully " << std::endl;

	get_mat_pos();

	std::cout << "Class constructed successfully :)" << std::endl;

}


// Destructor

sndMillepedeCaller::~sndMillepedeCaller()
{

	delete m_gbl_mille_binary;
	m_output_file->cd();
	m_output_tree->Write();
	m_output_file->Write();
	m_output_file->Close();

}


// Used for reading txt file 

std::vector<std::string> split(const std::string& input, char delimiter) 
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(input);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}




// Read mat info from txt file and fill the m_nominal_mat maps
/* 
   m_nominal_mat_centerpos -> <string,TVector3>
   						   -> the keys are the mat names (like V1VM2)
   						   -> corresponding value is the nominal geometric center position of the mat 
*/

void sndMillepedeCaller::get_mat_pos()
{
	VectorTriple mat_info;
	TVector3 mat_center;
	TVector3 mat_top, mat_bot;

	// Read mat info from file
	std::ifstream file_read("cpp_read_info.txt");

	std::string file_str;


	while (std::getline(file_read, file_str))
	{
    	// Process str
    	std::vector<std::string> parts = split(file_str, ' ');
		if (parts.size() != 10) {
			throw std::runtime_error("Invalid input format!");
		}
		float xxx = std::stof(parts[1]);
		float yyy = std::stof(parts[2]);
		float zzz = std::stof(parts[3]);
		float z1z1 = std::stof(parts[4]);
		float z2z2 = std::stof(parts[5]);
		float x1x1 = std::stof(parts[6]);
		float x2x2 = std::stof(parts[7]);
		float y1y1 = std::stof(parts[8]);
		float y2y2 = std::stof(parts[9]);
		int det_id = std::stof(parts[0]);

		// Create a unique key for the mat based on its ID
		std::string mat_key = formatMatKey(det_id);


		// Assign position to the corresponding mat
		m_nominal_mat_centerpos[mat_key] = TVector3(xxx,yyy,zzz);
		m_nominal_mat_toppos[mat_key] = TVector3(x2x2,y2y2,z2z2);
		m_nominal_mat_botpos[mat_key] = TVector3(x1x1,y1y1,z1z1);

	}


}



// Form keys for mats using their detector id 
std::string sndMillepedeCaller::formatMatKey(int det_id) {
  string key = "V";
  key += to_string(det_id / 1000000);  // Extract first digit

  int second_digit = (det_id % 1000000) / 100000;
  key += (second_digit == 0) ? "HM" : "VM";  // Add "HM" or "VM" based on second digit

  int third_digit = (det_id % 10000) / 1000;  // Extract third digit (minus 1 since the ids don't match in the geofile for some reason)
  key += to_string(third_digit);

  return key;
}




// 

std::vector<gbl::GblPoint> sndMillepedeCaller::list_hits(const GBL_seed_track* track, double sigma_spatial, map<int,double>* pede_corrections, TTree* tree)
{

	vector<TVector3> linear_model = {track->get_position(), track->get_direction()};
	vector<gbl::GblPoint> result_gbl_points = {};

	vector<pair<int,double>> points_from_track = track->get_hits();
	size_t n_track_points = points_from_track.size();
	result_gbl_points.reserve(n_track_points);

	//

	TVector3 PCA_track(0,0,0);
	TVector3 PCA_fiber(0,0,0);
	TVector3 closest_approach(0,0,0);

    std::cout << "HERE 1" << std::endl;

	int det_id_hit, driftside;
	double track_distance, rt, residual;

	for(size_t i = 0; i < n_track_points; ++i)
	{

        std::cout << "HERE 2" << std::endl;

		TVector3 PCA_track_last = PCA_track;
		pair<int, double> point = points_from_track[i];

		det_id_hit = point.first;

        std::cout << "HERE 3" << std::endl;

		// Determine the mat key based on det_id
		std::string formatted_string = formatMatKey(det_id_hit); 
		std::string mat_key = formatted_string;

        std::cout << "MAT_KEY IS" << mat_key << std::endl;

		//if (m_nominal_mat_centerpos.count(mat_key) > 0)
		//{

		// Retrieve positions from get_mat_pos()
		TVector3 vbot = m_nominal_mat_botpos[mat_key];
		TVector3 vtop = m_nominal_mat_toppos[mat_key];
		TVector3 mat_center = m_nominal_mat_centerpos[mat_key];

        std::cout << "HERE 4" << std::endl;

		if (pede_corrections)
		{
			vector<int> labels_for_mat = calc_labels(det_id_hit);
			double correction_x = (*pede_corrections)[labels_for_mat[0]];
			double correction_y = (*pede_corrections)[labels_for_mat[1]];
			//double correction_z = (*pede_corrections)[labels_for_mat[2]];
			double correction_z = 0;

			// Apply translation
			vbot[0] = vbot[0] + correction_x;
			vtop[0] = vtop[0] + correction_x;

			vbot[1] = vbot[1] + correction_y;
			vtop[1] = vtop[1] + correction_y;

			vbot[2] = vbot[2] + correction_z;
			vtop[2] = vtop[2] + correction_z;

			double rotation_gamma = (*pede_corrections)[labels_for_mat[5]];

			// Apply rotation
			TRotation rot;
			rot.RotateZ(rotation_gamma);

			string mat_name = m_channel_id_to_mat[point.first];
			TVector3 m_center = m_nominal_mat_centerpos[mat_name];
			TVector3 m_top_new = m_center + (rot * (vtop - m_center));
			TVector3 m_bot_new = m_center + (rot * (vbot - m_center));

			vtop = m_top_new;
			vbot = m_bot_new; 
		}


		double measurement = point.second; // rt distance (cm)

		closest_approach = calc_shortest_distance(vtop, vbot, linear_model[0], linear_model[1], &PCA_fiber, &PCA_track);

		TMatrixD* jacobian;

		if (i != 0)
		{
			jacobian = calc_jacobian(PCA_track_last, PCA_track);
		}
		else
		{
			jacobian = new TMatrixD(5,5);
			jacobian -> UnitMatrix();
		}

	
		result_gbl_points.push_back(gbl::GblPoint(*jacobian));
		add_measurement_info(result_gbl_points.back(), closest_approach, measurement, sigma_spatial);


	
		// Calculate labels and global derivatives for a hit
		vector<int> label = calc_labels(point.first);

		TVector3 fiber_bot_to_top = vtop - vbot;

		TVector3 measurement_prediction, alignment_origin;
		string module_descriptor;

		module_descriptor = m_channel_id_to_mat[point.first];
		alignment_origin = m_nominal_mat_centerpos[mat_key];

		measurement_prediction = PCA_track - alignment_origin;

		TMatrixD* globals = calc_global_parameters(measurement_prediction, closest_approach, linear_model, fiber_bot_to_top);
		result_gbl_points.back().addGlobals(label, *globals);

		delete globals;
		delete jacobian;

		#pragma omp critical
		{
			if (tree)
			{
				tree->SetBranchAddress("detectorID", &det_id_hit);
				tree->SetBranchAddress("driftside", &driftside);
				tree->SetBranchAddress("trackDistance", &track_distance);
				tree->SetBranchAddress("rt", &rt);
				tree->SetBranchAddress("residual", &residual);
				tree->SetBranchAddress("fiber_pca_x", &PCA_fiber[0]);
				tree->SetBranchAddress("fiber_pca_y", &PCA_fiber[1]);
				tree->SetBranchAddress("fiber_pca_z", &PCA_fiber[2]);
				tree->SetBranchAddress("meas_x", &closest_approach[0]);
				tree->SetBranchAddress("meas_y", &closest_approach[1]);
				tree->SetBranchAddress("meas_z", &closest_approach[2]);
				det_id_hit = point.first;
				driftside = closest_approach[0] < 0 ? -1 : 1;
				track_distance = closest_approach.Mag();
				rt = measurement;
				residual = measurement - closest_approach.Mag();
				tree->Fill();
			}
		}

//		}
		// else
		// {
		// 	cout << "Error retrieving mat positions!" << endl;
		// }

	}

	return result_gbl_points;

}





// Calculate global parameters

TMatrixD* sndMillepedeCaller::calc_global_parameters(const TVector3& measurement_prediction, const TVector3& closest_approach, const vector<TVector3>& linear_model, const TVector3& fiber_bot_to_top) const
{

	TRotation global_to_alignment;
	global_to_alignment.SetYAxis(fiber_bot_to_top);
	global_to_alignment.Invert();

	TMatrixD matD_global_to_alignment = TRot_to_TMatrixD(global_to_alignment); //debugging

	TVector3 measurement_prediction_in_alignment_system = global_to_alignment * measurement_prediction;

	TVector3 track_direction_in_alignment_system = global_to_alignment * linear_model[1];



	TRotation global_to_measurement;
	global_to_measurement.SetXAxis(closest_approach);
	global_to_measurement.Invert();

	TMatrixD matD_global_to_measurement = TRot_to_TMatrixD(global_to_measurement); // debugging

	TVector3 measurement_prediction_in_measurement_system = global_to_measurement * measurement_prediction;



	TRotation alignment_to_measurement = global_to_measurement * global_to_alignment.Inverse();
	alignment_to_measurement.Invert();

	TMatrixD matD_alignment_to_measurement = TRot_to_TMatrixD(alignment_to_measurement); // debugging



	// dmdg: matrix representing the derivatives of the distorted measurment wrt the global parameters 
	//		 dimensions: 3x6 for a rigid body alignment 
	//		 shows the relation between residuals and global parameters, which will be provided to GBL points

	// drdm: matrix representing the derivatives of the measurement wrt the track parameters


	TMatrixD dmdg(3,6);
	TMatrixD* result = new TMatrixD(3,6);

	TMatrixD drdm(3,3);
	drdm.UnitMatrix();

	
	TVector3 nominal_measurementplane_normal(0,0,1);

	double scalar_prod = track_direction_in_alignment_system.Dot(nominal_measurementplane_normal);

	for(short i = 0; i < drdm.GetNrows(); ++i)
	{
		for(short j = 0; j < drdm.GetNcols(); ++j)
		{
			drdm[i][j] -= track_direction_in_alignment_system[i] * nominal_measurementplane_normal[j] / scalar_prod;

		}
	}


	dmdg.Zero();
	result->Zero();



	// Fill the dmdg matrix
	/* 

	dmdg = | 1  0  0   0   z  -y |
		   | 0  1  0  -z   0   x |	
		   | 0  0  1   y  -x   0 |

	*/

	dmdg[0][0] = dmdg[1][1] = dmdg[2][2] = 1;
	dmdg[0][4] = measurement_prediction_in_alignment_system[2];
	dmdg[0][5] = - measurement_prediction_in_alignment_system[1];
	dmdg[1][3] = - measurement_prediction_in_alignment_system[2];
	dmdg[1][5] = measurement_prediction_in_alignment_system[0];
	dmdg[2][3] = measurement_prediction_in_alignment_system[1];
	dmdg[2][4] = - measurement_prediction_in_alignment_system[0];


	TMatrixD global_derivatives_in_alignment_system(3,6);
	global_derivatives_in_alignment_system.Mult(drdm, dmdg);

	result->Mult(matD_alignment_to_measurement, global_derivatives_in_alignment_system);

	return result;

}



gbl::GblTrajectory sndMillepedeCaller::perform_GBL_refit(const GBL_seed_track& track, double sigma_spatial, map<int, double>* pede_corrections, const char* spillname)
{

	vector<gbl::GblPoint> points = list_hits(&track, sigma_spatial, pede_corrections, m_output_tree);
	gbl::GblTrajectory traj(points, false); // False since no magnetic field

	traj.milleOut(*m_gbl_mille_binary);

	// Check if the GBL trajectory is valid or not
	if(!traj.isValid())
	{
		cout << "ERROR! Invalid GBL trajectory." << endl;
		throw runtime_error("GBL track is invalid!");
	}

	int rc, ndf;
	double chi2, lostWeight;

	cout << "~~~~~~~~~ PERFORMING GBL REFIT ~~~~~~~~~" << endl;

	rc = traj.fit(chi2, ndf, lostWeight);

	cout << "GBL Refit chi2: " << chi2 << endl;
	cout << "GBL Refit ndf: " << ndf << endl;

	// Calculate the probability for chi2 and number of degrees of freedom (ndf)
	// Calculations are based on the incomplete gamma function P(a,x), where a=ndf/2 and x=chi2/2

	cout << "Prob: " << TMath::Prob(chi2, ndf) << endl;

	print_fitted_residuals(traj);
	print_fitted_track(traj);

	return traj;

}




/**
 * Convert a TRotation to a its 3x3 rotation matrix given as TMatrixD
 *
 * @brief Convert a TRotation to a (3x3) TMatrixD
 *
 * @author Stefan Bieschke
 * @date Sep. 09, 2019
 * @version 1.0
 *
 * @param rot TRotation object for that the rotation matrix is needed
 *
 * @return TMatrixD object with dimensions 3x3
 */
TMatrixD sndMillepedeCaller::TRot_to_TMatrixD(const TRotation& rot) const
{
	TMatrixD result(3,3);
	for(uint8_t i = 0; i < 3; ++i)
	{
		for(uint8_t j = 0; j < 3; ++j)
		{
			result[i][j] = rot[i][j];
		}
	}
	return result;
}




/**
 * Calculate the rotation matrix in a way such that a given vector v is the x-axis of the rotated coordinate frame.
 *
 * @brief Rotation of a given vector in lab frame so that it is new x axis
 *
 * @author Stefan Bieschke
 * @date Aug. 09, 2019
 * @version 1.0
 *
 * @param v TVector3 that is meant to be the new x axis of the rotated coordinate frame
 *
 * @return TRotation containing the rotation matrix
 */
TRotation sndMillepedeCaller::calc_rotation_of_vector(const TVector3& v) const
{
	TRotation rot;
	rot.SetXAxis(v);

	return rot;
}





/**
 * Calculate the projection matrix (dimension 2x2) from the global reference frame (x,y,z) to the measurement system (u,v,w) for an
 * individual hit.
 *
 * @brief Calculate projection matrix global to measumrent system
 *
 * @author Stefan Bieschke
 * @date Oct. 07, 2019
 * @version 1.0
 *
 * @param fit_system_base_vectors Matrix with dimensions 2x3 with the base vectors of the local fit system expressed in global system. E.g [(1., 0., 0.),(0.,1.,0.)] for x,y and track in general z direction.
 * @param rotation_global_to_measurement Rotation matrix rotating the global frame to the measurement frame
 *
 * @return TMatrixD (dimension 2x2) containing the projection of the fit system (x,y) on the measurement system
 */
TMatrixD sndMillepedeCaller::calc_projection_matrix(
		const TMatrixD& fit_system_base_vectors,
		const TMatrixD& rotation_global_to_measurement) const
{
	TMatrixD result(2,2); //projection matrix has dimensions 2x2
	TMatrixD measurement_to_global(rotation_global_to_measurement); //copy rotational matrix
	measurement_to_global.Invert(); //invert matrix in place
	measurement_to_global.ResizeTo(3,2); //TODO check if this is correct, want to skip column normal to measurement direction
	result.Mult(fit_system_base_vectors,measurement_to_global);
	result.Invert();
	return result;
}





// Calculate labels 

vector<int> sndMillepedeCaller::calc_labels(const int det_id) const
{
	vector<int> labels;

    // Extract components from det_id
    /* det_id format: 1234567
    				  1: volume (1-5)
    				  2: orientation (0:H, 1:V)
    				  3: mat # (0-2)
    				  4: sipm # (0-3)
    				  567: channel # (0-127)
	   e.g. det_id 3120097 refers to the channel 97 of the 1st sipm of vertical mat 2 of volume 3
    */
    int channel = det_id % 1000;
    int sipm = (det_id / 1000) % 10;
    int mat = (det_id / 10000) % 10;
    int orientation = (det_id / 100000) % 10;
    int station = (det_id / 1000000) % 10;

    // Define labels for each alignable unit
    /* Label format: ABCX
    				 A: volume (1-5)
    				 B: orientation (0:H, 1:V)
    				 C: mat (0-2)
    				 D: global parameter (1-6)

		For each alignable unit (i.e. mat), there are 6 labels, 1 for each global parameter.
		For all mats, we have a total of 180 labels. 

		e.g. label 3024 represents the rotation of horizontal mat 3 of scifi volume 3, around x-axis
    */
    for (int unit = 1; unit <= 6; ++unit) {
        int label = station * 1000 + orientation * 100 + mat * 10 + unit;
        labels.push_back(label);
    }

    return labels;
}






/* Calculate the closest approach vector, which is the shortest distance from sense fiber to the track.
   Closest approach vector is perpendicular to both track and fiber. 

 * @author Stefan Bieschke
 * @date Oct. 01, 2019
 * @version 1.1
 *
 * @param fiber_top Top position of the fiber as TVector3 with x,y,z components
 * @param fiber_bot Bottom position of the fiber as TVector3 with x,y,z components
 * @param track_pos Some position on the track. Could be anything but must be on the (straight) track or track segment
 * @param track_mom Momentum vector of the (straight) track or track segment
 * @param PCA_on_fiber Coordinates of point of closest approach (PCA) on the fiber (defaults to nullptr)
 * @param PCA_on_track Coordinates of point of closest approach (PCA) on the track (defaults to nullptr)
 *
 * @return TVector3 of shortest distance pointing from the fiber to the track

*/

TVector3 sndMillepedeCaller::calc_shortest_distance(const TVector3& fiber_top,
		const TVector3& fiber_bot, const TVector3& track_pos,
		const TVector3& track_mom, TVector3* PCA_on_fiber,
		TVector3* PCA_on_track) const
{

	TVector3 fiber_dir = fiber_top - fiber_bot; // fiber direction

	/* Construct a helper plane to simplify the problem of finding the shortest distance between a fiber and a straight track.
	It allows us to reduce the problem to a two-dimensional space instead of working in three dimensions directly.
	
	Helper plane is defined by a position vector (plane_pos) and two direction vectors (plane_dir_1 and plane_dir_2).
	plane_dir_1 is parallel to the track, and plane_dir_2 is parallel to the fiber.
	
	It contains one of the straight lines (either the fiber or the track) and is parallel to the other one.
	By working in the plane, we can simplify the problem of finding the shortest distance between the wire and the track to finding the shortest distance between two lines in 2D space.
	
	The plane helps us set up an equation system (M * x = c) to find the point of closest approach (PCA) between the fiber and the track.
	The equation system involves coefficients (elements of matrix M) and constants (elements of vector c) derived from the geometry of the fiber, track, and the constructed plane.

	After finding the PCA points in the plane, we can translate them back to 3D space to calculate the shortest distance vector.
	*/
	TVector3 plane_pos = track_pos - fiber_bot;
	TVector3 plane_dir_1(track_mom);
	TVector3 plane_dir_2(-1 * fiber_dir);

	// Construct components of equation system M * x = c
	TVectorD const_vector(2);
	TMatrixD coeff_matrix(2,2);

	const_vector[0] = -(plane_pos.Dot(track_mom));
	const_vector[1] = -(plane_pos.Dot(fiber_dir));

	coeff_matrix[0][0] = plane_dir_1.Dot(track_mom);
	coeff_matrix[0][1] = plane_dir_2.Dot(track_mom);
	coeff_matrix[1][0] = plane_dir_1.Dot(fiber_dir);
	coeff_matrix[1][1] = plane_dir_2.Dot(fiber_dir);

	/* TDecompLU performs LU decomposition on a given matrix, where the matrix is decomposed into the product of a lower triangular matrix (L) and an upper triangular matrix (U).
       Once the matrix is decomposed, TDecompLU can efficiently solve systems of linear equations of the form A * x = b, where A is the original matrix and b is a given vector.
	*/
	TDecompLU solvable_matrix(coeff_matrix);
	TVectorD result(const_vector);
	int rc = solvable_matrix.Solve(result);

	TVector3 PCA_track(track_pos + result[0] * track_mom); // point of closest approach (PCA) on track
	TVector3 PCA_fiber(fiber_bot + result[1] * fiber_dir); // point of closest approach on fiber

	// Option to store the coordinates of the PCAs on the fiber and the track if pointers to TVector3 objects are provided
	if (PCA_on_fiber)
	{
		*(PCA_on_fiber) = PCA_fiber;
	}
	if (PCA_on_track)
	{
		*(PCA_on_track) = PCA_track;
	}

	return TVector3(PCA_track - PCA_fiber); // Vector of closest approach

}




/**
 * Calculate a position and a direction vector for a linear model that models a given track from genfit. While the track
 * from genfit might take scattering in detector material into account, this linear model doesn't consider scatterers.
 * The way this works is to calculate and return a straight track between the first and the last hit of the passed track.
 *
 * @brief Linear model of the passed track
 *
 * @author Stefan Bieschke
 * @date Sep. 09, 2019
 * @version 1.0
 *
 * @param track Track that is meant to be modeled
 *
 * @return std::vector of length two, whos first entry is the position vector, second one being the direction vector
 */
vector<TVector3> sndMillepedeCaller::linear_model_wo_scatter(const genfit::Track& track) const
{
	vector<TVector3> result(2);

	//get first and last fitted states
	size_t n_hits = track.getNumPointsWithMeasurement();
	genfit::StateOnPlane first_hit = track.getFittedState(0);
	genfit::StateOnPlane last_hit = track.getFittedState(n_hits - 1);

	//position is fitted position on first hit
	TVector3 pos = first_hit.getPos();

	//direction is difference between positions at first and last hits
	TVector3 dir = last_hit.getPos() - first_hit.getPos();
	result[0] = pos;
	result[1] = dir;

	return result;
}


void sndMillepedeCaller::print_model_parameters(const vector<TVector3>& model) const
{
	double slope_x = model[1].X() / model[1].Z();
	double slope_y = model[1].Y() / model[1].Z();

	cout << "Printing linear track model parameters" << endl;
	cout << "Initial fit position:" << endl;
	model[0].Print();
	cout << "Track direction:" << endl;
	model[1].Print();
	cout << "Model parameters: x0, y0, slope x, slope y" << endl;
	cout << "(" << model[0].X() << ", " << model[0].Y() << ", " << slope_x  << ", " << slope_y <<")" << endl;
}



/**
 * Calculate the jacobi matrix for a linear model from the points of closest approach on the track for two consecutive hits.
 *
 * @brief Calculate jacobian between two consecutive hits
 *
 * @author Stefan Bieschke
 * @date November 22, 2019
 * @version 1.0
 *
 * @param PCA_1 Point of closest approach on the track for the first hit
 * @param PCA_2 Point of closest approach on the track for the second hit
 *
 * @return Pointer to a heap object of TMatrixD type with dimensions 5x5
 *
 * @warning Heap object without auto deletion
 */
TMatrixD* sndMillepedeCaller::calc_jacobian(const TVector3& PCA_1, const TVector3& PCA_2) const
{
	TMatrixD* jacobian = new TMatrixD(5,5);

	// 1.) init unity matrix
	jacobian->UnitMatrix();

	double dz = PCA_2.Z() - PCA_1.Z();

	//2.) enter non-zero off diagonal elements
	(*jacobian)[3][1] = dz;
	(*jacobian)[4][2] = dz;

	return jacobian;
}





// Calculate the measurement information and add it to a gbl::GblPoint object 
// Which involves transforming the measurement system to the fit system, calculating the rotated residual, defining the precision, and then adding the measurement to the GBL point. 

void sndMillepedeCaller::add_measurement_info(gbl::GblPoint& point, const TVector3& closest_approach, const double measurement, const double sigma_spatial) const
{
	//define projection of measurement system to the fit system
	TMatrixD fit_system_base_vectors(2,3);
	fit_system_base_vectors.Zero();
	fit_system_base_vectors[0][0] = 1.0; 	//first row vector for x direction
	fit_system_base_vectors[1][1] = 1.0; 	//second row vector for y direction

	TRotation rotation_global_to_measurement = calc_rotation_of_vector(closest_approach);
	rotation_global_to_measurement.Invert();
	TMatrixD rot_mat = TRot_to_TMatrixD(rotation_global_to_measurement);
	TMatrixD projection_matrix = calc_projection_matrix(fit_system_base_vectors,rot_mat);
	TVectorD rotated_residual(2);
	rotated_residual[0] = measurement - closest_approach.Mag();
	rotated_residual[1] = 0;

	TVectorD precision(rotated_residual);
	precision[0] = 1.0 / TMath::Power(sigma_spatial,2);
	point.addMeasurement(projection_matrix,rotated_residual,precision);
}




// Create output tree for debugging

TTree* sndMillepedeCaller::create_output_tree()
{

	TTree* tree = new TTree("DebuggingTree", "Tree for debug info");
	
	int det_id, driftside;
	double track_distance, rt, residual, fiber_pca_x, fiber_pca_y, fiber_pca_z, meas_x, meas_y, meas_z;
	
	tree->Branch("track_distance", &track_distance, "track_distance/D");
	tree->Branch("detectorID", &det_id, "detectorID/I");
	tree->Branch("driftside",&driftside,"driftside/I");
	tree->Branch("rt", &rt, "rt/D");
	tree->Branch("residual", &residual, "residual/I");
	tree->Branch("fiber_pca_x", &fiber_pca_x, "fiber_pca_x/D");
	tree->Branch("fiber_pca_y", &fiber_pca_y, "fiber_pca_y/D");
	tree->Branch("fiber_pca_z", &fiber_pca_z, "fiber_pca_z/D");
	tree->Branch("meas_x", &meas_x, "meas_x/D");
	tree->Branch("meas_y", &meas_y, "meas_y/D");
	tree->Branch("meas_z", &meas_z, "meas_z/D");

	return tree;
}





/**
 * Checks, if the points on a genfit seed track are ordered by arc length. The arc length
 * is the distance from an early point on the track, e.g the vertex or the very first hit.
 * This doesn't matter if the arc length itself is not used but the GblPoints must only
 * be ordered correctly.
 *
 * @brief Checks if points on seed track are ordered by arc length
 *
 * @author Stefan Bieschke
 * @date Nov. 6, 2019
 * @version 1.0
 *
 * @param track genfit seed track
 * @return true if points are correctly ordered, false else
 *
 * @note This is mainly for debugging and maintenance purposes, not recommended to be called in productive code
 */
bool sndMillepedeCaller::check_ordered_by_arclen(const genfit::Track& track) const
{
	size_t n_points = track.getNumPointsWithMeasurement();
	double z = 0.0;

	for(size_t i = 0; i < n_points; ++i)
	{
		double z_next = track.getFittedState(i).getPos().Z();
		if(z_next > z)
		{
			z = z_next;
		}
		else
		{
			return false;
		}
	}

	return true;
}


// For debugging 

void sndMillepedeCaller::print_seed_hits(const genfit::Track& track) const
{
	size_t n_hits = track.getNumPointsWithMeasurement();
	const vector< genfit::TrackPoint* > points = track.getPoints();

	for(size_t i = 0; i < n_hits; ++i)
	{
		int det_id = points[i]->getRawMeasurement()->getDetId();
		cout << "Hit: " << i << "\t" << "ID: " << det_id << endl;
	}
}


/**
 * Prints the residuals of the fitted trajectory to all the measurements, the GblTrajectory contains in the form of
 * GblPoint objects. This is outputted as text.
 *
 * @brief Text output of residuals for each hit of fitted GblTrajectory
 *
 * @author Stefan Bieschke
 * @date Jan. 24, 2020
 * @version 1.0
 *
 * @parameter trajectory GblTrajectory that is already fitted
 */
void sndMillepedeCaller::print_fitted_residuals(gbl::GblTrajectory& trajectory) const
{
	//print residuals
	TVectorD residuals(1);
	TVectorD meas_errors(1);
	TVectorD res_errors(1);
	TVectorD down_weights(1);
	unsigned int numRes;
	for (unsigned int j = 1; j <= trajectory.getNumPoints(); ++j)
	{
		trajectory.getMeasResults(j, numRes, residuals, meas_errors, res_errors,down_weights);
		cout << "Hit: " << j << " Residual: " << residuals[0] << endl;
	}
}



// For debugging 

void sndMillepedeCaller::print_fitted_track(gbl::GblTrajectory& trajectory) const
{
	TVectorT<double> parameters(5);
	TMatrixTSym<double> covariance(5, 5);
	for (unsigned int i = 1; i <= trajectory.getNumPoints(); ++i)
	{
		trajectory.getResults(i, parameters, covariance);
		cout << "Hit: " << i << endl;
		for (unsigned int j = 0; j < parameters.GetNrows(); ++j)
		{
			cout << "Parameter: " << j << " value: " << parameters[j] << endl;
		}
	}
}
