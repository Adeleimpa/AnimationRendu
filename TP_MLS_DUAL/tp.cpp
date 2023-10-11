// -------------------------------------------
// gMini : a minimal OpenGL/GLUT application
// for 3D graphics.
// Copyright (C) 2006-2008 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is dirty in the
// meaning that there is no attention paid to
// proper class attribute access, memory
// management or optimisation of any kind. It
// is designed for quick-and-dirty testing
// purpose.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"



// points blancs
std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

// points rouges
std::vector< Vec3 > positions2;
std::vector< Vec3 > normals2;

//BasicANNkdTree kdtree;

// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 640;
static unsigned int SCREENHEIGHT = 480;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;

struct Voxel {
    public:
        std::vector<int> index_corners;
        Vec3 centroid;
    
};



struct Grid {
    public: 
        std::vector<std::vector<std::vector<Vec3>>> points;
        std::vector<Vec3> normals;

        Vec3 min_corner;
        Vec3 max_corner;

        std::vector<double> scalars;
        std::vector<bool> isNegative;

        int resolution;
        double voxel_length;


    // generate the eight corners of a box described by its extreme corners (max_corner and min_corner)
    /*std::vector<Vec3> generateCorners(Vec3 max_corner, Vec3 min_corner){
        Vec3 TLF_corner(min_corner[0], max_corner[1], min_corner[2]); // Top Left Front
        Vec3 TLB_corner(min_corner[0], max_corner[1], max_corner[2]); // Top Left Back
        Vec3 TRF_corner(max_corner[0], max_corner[1], min_corner[2]); // Top Right Back
        Vec3 DLB_corner(min_corner[0], min_corner[1], max_corner[2]); // Down Left Front
        Vec3 DRF_corner(max_corner[0], min_corner[1], min_corner[2]); // Down Right Front
        Vec3 DRB_corner(max_corner[0], min_corner[1], max_corner[2]); // Down Right Back

        std::vector<Vec3> corners{max_corner, TRF_corner, TLF_corner, TLB_corner, DLB_corner, min_corner, DRF_corner, DRB_corner};

        // uncomment to sketch the cube
        glColor3f(3.0F, 3.0F, 3.0F); // white
        glBegin(GL_LINE_STRIP);
        for(unsigned int i = 0 ; i < corners.size(); i++) {
            glVertex3f( corners[i][0] , corners[i][1] , corners[i][2] );
        }
        glVertex3f(max_corner[0] , max_corner[1] , max_corner[2]);
        glVertex3f(TLB_corner[0] , TLB_corner[1] , TLB_corner[2]);
        glVertex3f(DLB_corner[0] , DLB_corner[1] , DLB_corner[2]);
        glVertex3f(DRB_corner[0] , DRB_corner[1] , DRB_corner[2]);
        glVertex3f(DRF_corner[0] , DRF_corner[1] , DRF_corner[2]);
        glVertex3f(TRF_corner[0] , TRF_corner[1] , TRF_corner[2]);
        glVertex3f(TLF_corner[0] , TLF_corner[1] , TLF_corner[2]);
        glVertex3f(min_corner[0] , min_corner[1] , min_corner[2]);
        
        glEnd();

        return corners;
    }*/

    // returns the center point of the bounding box described by max_corner and min_corner
    Vec3 getCenter(Vec3 max_corner, Vec3 min_corner){
        double center_x = min_corner[0] + (max_corner[0] - min_corner[0])/2;
        double center_y = min_corner[1] + (max_corner[1] - min_corner[1])/2;
        double center_z = min_corner[2] + (max_corner[2] - min_corner[2])/2;
        Vec3 center(center_x, center_y, center_z);
        return center;
    }

    bool checkScalarsInVoxel(int x , int y , int z, int index_pos){

        bool onePos = false;
        bool oneNeg = false;

        std::vector<double> scalars_in_voxel;
        scalars_in_voxel.push_back(scalars[index_pos]); // TLF
        if(z!=resolution) scalars_in_voxel.push_back(scalars[index_pos+1]); // TLB
        if(y!=resolution) scalars_in_voxel.push_back(scalars[index_pos+(resolution+1)]); // TRF
        if(z!=resolution && y!=resolution) scalars_in_voxel.push_back(scalars[index_pos+(resolution+1)+1]); // MAX
        if(x!=resolution) scalars_in_voxel.push_back(scalars[index_pos+( (resolution+1) * (resolution+1) )]); // MIN
        if(x!=resolution && z!=resolution) scalars_in_voxel.push_back(scalars[index_pos+( (resolution+1) * (resolution+1) ) +1]); // DLB
        if(x!=resolution && y!=resolution) scalars_in_voxel.push_back(scalars[index_pos+( (resolution+1) * (resolution+1) ) + (resolution+1)]); // DRF
        if(x!=resolution && y!=resolution && z!=resolution) scalars_in_voxel.push_back(scalars[index_pos+( (resolution+1) * (resolution+1) ) + (resolution+1) +1]); // DRB

        for(unsigned int it = 0; it < scalars_in_voxel.size(); it++){
            if(scalars_in_voxel[it] < 0){
                oneNeg = true;
            }else{
                onePos = true;
            }

            if(onePos && oneNeg){
                return true;
            }
        }
        
        return false;
    }


    void buildGrid(Vec3 center, float length, int reso){
        resolution = reso;

        //intialize points vector
        points = std::vector<std::vector<std::vector<Vec3>>>(resolution+1, std::vector<std::vector<Vec3>>(resolution+1, std::vector<Vec3>(resolution+1)));

    	min_corner = Vec3(center[0]-(length/2), center[1]-(length/2), center[2]-(length/2));
        max_corner = Vec3(center[0]+(length/2), center[1]+(length/2), center[2]+(length/2));

        double sq_dim = max_corner[0] - min_corner[0];
        voxel_length = sq_dim/(double)resolution;

        Vec3 current_min_corner = min_corner;
        Vec3 current_max_corner;
        for(int i = 0; i <= resolution; i++){ // x-axis
            for(int j = 0; j <= resolution; j++){ // y-axis
                for(int k = 0; k <= resolution; k++){ // z-axis
                    // update current_min_corner and current_max_corner
                    current_min_corner[0] = min_corner[0] + i*voxel_length;
                    current_min_corner[1] = min_corner[1] + j*voxel_length;
                    current_min_corner[2] = min_corner[2] + k*voxel_length;

                    current_max_corner = current_min_corner + Vec3(voxel_length, voxel_length, voxel_length);

                    // ----------CREATE POINT----------
                    // add only Top Left Front corner
                    Vec3 TLF_corner(current_min_corner[0], current_max_corner[1], current_min_corner[2]);
                    glBegin(GL_POINTS); // sketch point
                        glVertex3f(TLF_corner[0], TLF_corner[1], TLF_corner[2]);
                    glEnd();
                    points[i][j][k] = TLF_corner;
                }
            }
        }
    }
};


// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN (const std::string & filename , std::vector< Vec3 > & o_positions , std::vector< Vec3 > & o_normals ) {
    unsigned int surfelSize = 6;
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    o_positions.clear ();
    o_normals.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back (Vec3 (pn[i], pn[i+1], pn[i+2]));
            o_normals.push_back (Vec3 (pn[i+3], pn[i+4], pn[i+5]));
        }

        if (numOfPoints < surfelSize*READ_BUFFER_SIZE) break;
    }
    fclose (in);
    delete [] pn;
}
void savePN (const std::string & filename , std::vector< Vec3 > const & o_positions , std::vector< Vec3 > const & o_normals ) {
    if ( o_positions.size() != o_normals.size() ) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE * outfile = fopen (filename.c_str (), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for(unsigned int pIt = 0 ; pIt < o_positions.size() ; ++pIt) {
        fwrite (&(o_positions[pIt]) , sizeof(float), 3, outfile);
        fwrite (&(o_normals[pIt]) , sizeof(float), 3, outfile);
    }
    fclose (outfile);
}
void scaleAndCenter( std::vector< Vec3 > & io_positions ) {
    Vec3 bboxMin( FLT_MAX , FLT_MAX , FLT_MAX );
    Vec3 bboxMax( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , io_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , io_positions[pIt][coord] );
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>( bboxMax[0]-bboxMin[0] , std::max<float>( bboxMax[1]-bboxMin[1] , bboxMax[2]-bboxMin[2] ) );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}

void applyRandomRigidTransformation( std::vector< Vec3 > & io_positions , std::vector< Vec3 > & io_normals ) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}

void subsample( std::vector< Vec3 > & i_positions , std::vector< Vec3 > & i_normals , float minimumAmount = 0.1f , float maximumAmount = 0.2f ) {
    std::vector< Vec3 > newPos , newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for( unsigned int i = 0 ; i < indices.size() ; ++i ) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin() , indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount-minimumAmount)*(float)(rand()) / (float)(RAND_MAX));
    newPos.resize( newSize );
    newNormals.resize( newSize );
    for( unsigned int i = 0 ; i < newPos.size() ; ++i ) {
        newPos[i] = i_positions[ indices[i] ];
        newNormals[i] = i_normals[ indices[i] ];
    }
    i_positions = newPos;
    i_normals = newNormals;
}

bool save( const std::string & filename , std::vector< Vec3 > & vertices , std::vector< unsigned int > & triangles ) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size() , n_triangles = triangles.size()/3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << triangles[3*f] << " " << triangles[3*f+1] << " " << triangles[3*f+2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}




// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}



void drawTriangleMesh( std::vector< Vec3 > const & i_positions , std::vector< unsigned int > const & i_triangles ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_triangles.size() / 3 ; ++tIt) {
        Vec3 p0 = i_positions[3*tIt];
        Vec3 p1 = i_positions[3*tIt+1];
        Vec3 p2 = i_positions[3*tIt+2];
        Vec3 n = Vec3::cross(p1-p0 , p2-p0);
        n.normalize();
        glNormal3f( n[0] , n[1] , n[2] );
        glVertex3f( p0[0] , p0[1] , p0[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();
}

void drawPointSet( std::vector< Vec3 > const & i_positions , std::vector< Vec3 > const & i_normals ) {
    glBegin(GL_POINTS);
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        glNormal3f( i_normals[pIt][0] , i_normals[pIt][1] , i_normals[pIt][2] );
        glVertex3f( i_positions[pIt][0] , i_positions[pIt][1] , i_positions[pIt][2] );
    }
    glEnd();
}

float Gaussian(float r, float d){ // r = la largeur de bande (bandwidth) 
    // Plus un voisin est proche, plus son poids est élevé.

    float d_squared = d * d;
    float r_squared = r * r;
    return exp(-d_squared/r_squared);
}

float Wendland(float r, float d){
    // attribue un poids en fonction de la distance radiale du voisin

    float a = pow(1 - (d/r),4);
    float b = 1 + 4 * (d/r);
    return a*b;
}

float Singular(float r, float d){
    // attribue un poids inversement proportionnel à la distance radiale. 
    // Plus la distance est grande, moins le poids est élevé.

    return pow(r/d, 2);
}


Vec3 project(Vec3 const &inputPoint, Vec3 const &centroid, Vec3 const &c_normal){
    float dot_prod = Vec3::dot(inputPoint - centroid, c_normal);
    Vec3 res = inputPoint - (dot_prod * c_normal);
    return res;
}

// X' = HPSS(X)
// X input point, X' output point
void HPSS(Vec3 inputPoint, Vec3 &outputPoint, Vec3 &outputNormal, std::vector<Vec3> const &positions, 
    std::vector<Vec3> const &normals, BasicANNkdTree const &kdtree, int kernel_type, float radius,
    unsigned int nbIterations = 5, unsigned int k = 20){

    // find the k nearest neighbors of inputPoint
    ANNidxArray id_nearest_neighbors = new ANNidx[ k ];
    ANNdistArray square_distances_to_neighbors = new ANNdist[ k ];

    // A chaque itération, on cherche les plus proches voisins, et on définit des poids vis a vis d’eux
    for (unsigned int i = 0; i < nbIterations; i++){
        kdtree.knearest(inputPoint, k , id_nearest_neighbors , square_distances_to_neighbors );

        // Calculate the median distance among neighbors
        std::vector<float> distances;
        for (unsigned int j = 0; j < k; j++) {
            distances.push_back(sqrt(square_distances_to_neighbors[j]));
        }
        std::sort(distances.begin(), distances.end());
        float median_distance = distances[k / 2];

        // compute centroid point and its normal
        Vec3 avg_neighbor_p = Vec3(0.,0.,0.);
        Vec3 avg_neighbor_n = Vec3(0.,0.,0.);

        float w; // weight
        float r = median_distance; // constant that adjusts itself
        float d;

        float weight_sum = 0.0;

        for (unsigned int i = 0; i < k; i++){

            d = sqrt(square_distances_to_neighbors[i]); // distance euclidienne entre le input point et son voisin i

            switch(kernel_type){
                case 0: w = Gaussian(r, d);
                case 1: w = Wendland(r, d);
                case 2: w = Singular(r, d);
                default: w = 1.0;
            }

            weight_sum += w;

            avg_neighbor_p += w * positions[id_nearest_neighbors[i]];
            avg_neighbor_n += w * normals[id_nearest_neighbors[i]];
        }
        // On trouve le plan qui passe au mieux par ces points (ACP pondérée)
        avg_neighbor_p /= (float) weight_sum;
        avg_neighbor_n /= (float) weight_sum;

        // compute output point by projecting the input point on its plane
        outputPoint = project(inputPoint, avg_neighbor_p, avg_neighbor_n); // x' = project(x,c,n)
        outputNormal = avg_neighbor_n;
        outputNormal.normalize();

        //outputNormal *= 0.2; // add noise

        inputPoint = outputPoint;
    }


    delete [] id_nearest_neighbors;
    delete [] square_distances_to_neighbors;
    
}

void dualContouring(BasicANNkdTree const &kdtree){

    Grid grid;
    grid.buildGrid(Vec3(0.0,0.0,0.0), 1.5, 32);

    // make a copy of grid points
    std::vector<Vec3> positionsIn, positionsInCopy;
    for (unsigned int i=0; i<grid.points.size(); i++){
        for (unsigned int j=0; j<grid.points[i].size(); j++){
            for (unsigned int k=0; k<grid.points[i][j].size(); k++){
                positionsIn.push_back(grid.points[i][j][k]);
                positionsInCopy.push_back(grid.points[i][j][k]);
                //std::cout << " Vec3 (" << grid.points[i][j][k][0] << ", " << grid.points[i][j][k][1] << ", " << grid.points[i][j][k][2] << ")" << std::endl;
            }
            //std::cout << "\n" << std::endl;
        }
        //std::cout << "\n" << std::endl;
    }

    // call HPSS and compute project of each point
    grid.normals.resize(positionsIn.size());
    positions2.clear();
    normals2.clear();

    for(unsigned int i = 0; i < positionsIn.size(); i++){
        HPSS(positionsIn[i], positionsIn[i], grid.normals[i], positions, normals, kdtree, 0, 1.0);
        double implicitFunction = Vec3::dot((positionsInCopy[i] - positionsIn[i]),grid.normals[i]); 
        grid.scalars.push_back(implicitFunction);
        if(grid.scalars[i]<0){grid.isNegative.push_back(true);}

        // to sketch it
        positions2.push_back(positionsIn[i]);
        normals2.push_back(positionsIn[i]);
    }


    // parcourir grille et checker si deux scalaires ont des signes opposés
    // si oui, on rajoute un point au centre de la cellule
    std::vector<Voxel> voxels; // only those that have a neg-pos pattern
    int index_pos = 0;
    for (unsigned int i=0; i<grid.points.size(); i++){
        for (unsigned int j=0; j<grid.points[i].size(); j++){
            for (unsigned int k=0; k<grid.points[i][j].size(); k++){

                bool negAndPos = grid.checkScalarsInVoxel(i, j, k, index_pos);

                if(negAndPos && i+1<=grid.resolution && j+1<=grid.resolution && k+1<=grid.resolution){
                    Voxel vox;
                    // TODO vox.index_corners.push_back()
                    vox.centroid = grid.getCenter(grid.points[i+1][j][k], grid.points[i][j+1][k+1]);
                    voxels.push_back(vox);
                }
                index_pos++;
            }
        }
    }
    //std::cout << voxels.size() << std::endl;


    // ----------------------------------------------------------------------------
    // TODO there is an error here, edges don't seem right

    // ensuite, on va parcourir toutes les arêtes de la grille (en x en y et en z)
    // arêtes en x
    for (int i = 0; i < ((grid.resolution+1) * (grid.resolution+1)); i++){
        float s1 = grid.scalars[i];
        float s2 = grid.scalars[i + (grid.resolution+1) * (grid.resolution+1)];
        float s3 = grid.scalars[2* (i + (grid.resolution+1) * (grid.resolution+1))];

        // si les deux extremités de l'arête ont un signe différent 
        if((s1 < 0 and s2 >= 0) or (s1 >= 0 and s2 < 0)){
            // TODO
            // alors on relie les 4 centres qui entourent l'arete
            // en créant deux triangles
            // on oriente les normales des triangles vers le coté positif de l'arete

            glColor3f(3.0F, 3.0F, 3.0F); // white
            glBegin(GL_LINE_STRIP);
            glVertex3f(positionsIn[i][0] , positionsIn[i][1] , positionsIn[i][2]);
            glVertex3f(positionsIn[i + (grid.resolution+1) * (grid.resolution+1)][0] , positionsIn[i + (grid.resolution+1) * (grid.resolution+1)][1] , positionsIn[i + (grid.resolution+1) * (grid.resolution+1)][2]);
            glEnd();
        }

        // si les deux extremités de l'arête ont un signe différent 
        if((s2 < 0 and s3 >= 0) or (s2 >= 0 and s3 < 0)){
            // TODO
            // alors on relie les 4 centres qui entourent l'arete
            // en créant deux triangles
            // on oriente les normales des triangles vers le coté positif de l'arete

            glBegin(GL_LINE_STRIP);
            glVertex3f(positionsIn[i + (grid.resolution+1) * (grid.resolution+1)][0] , positionsIn[i + (grid.resolution+1) * (grid.resolution+1)][1] , positionsIn[i + (grid.resolution+1) * (grid.resolution+1)][2]);
            glVertex3f(positionsIn[2* (i + (grid.resolution+1) * (grid.resolution+1))][0] , positionsIn[2* (i + (grid.resolution+1) * (grid.resolution+1))][1] , positionsIn[2* (i + (grid.resolution+1) * (grid.resolution+1))][2]);
            glEnd();
        }
    }

    // arêtes en y
    for(int i = 0; i < (grid.resolution+1) * (grid.resolution+1) * (grid.resolution+1) - (grid.resolution+1); i += ((grid.resolution+1)*(grid.resolution+1))){
        for(int j = 0; j < i + (grid.resolution+1) * (grid.resolution+1); j += (grid.resolution+1)){
            // on a une arête qui relie positionsIn[j] à positionsIn[j+(reso+1)]

            // si les deux extremités de l'arête ont un signe différent 
            if( (grid.scalars[j] < 0 and grid.scalars[j+(grid.resolution+1)] >= 0) or (grid.scalars[j+(grid.resolution+1)] < 0 and grid.scalars[j] >= 0) ){
                // TODO
                // alors on relie les 4 centres qui entourent l'arete
                // en créant deux triangles
                // on oriente les normales des triangles vers le coté positif de l'arete

                glBegin(GL_LINE_STRIP);
                glVertex3f(positionsIn[j][0] , positionsIn[j][1] , positionsIn[j][2]);
                glVertex3f(positionsIn[j+(grid.resolution+1)][0] , positionsIn[j+(grid.resolution+1)][1] , positionsIn[j+(grid.resolution+1)][2]);
                glEnd();
            }
        }
    }

    // arêtes en z
    for(int i = 0; i < positionsIn.size(); i += grid.resolution +1){

        for(int j = 0; j < i + grid.resolution; j++){
            // on a une arête qui relie positionsIn[j] à positionsIn[j+1]

            // si les deux extremités de l'arête ont un signe différent 
            if( (grid.scalars[j] < 0 and grid.scalars[j+1] >= 0) or (grid.scalars[j+1] < 0 and grid.scalars[j] >= 0) ){
                // TODO
                // alors on relie les 4 centres qui entourent l'arete
                // en créant deux triangles
                // on oriente les normales des triangles vers le coté positif de l'arete

                glBegin(GL_LINE_STRIP);
                glVertex3f(positionsIn[j][0] , positionsIn[j][1] , positionsIn[j][2]);
                glVertex3f(positionsIn[j+1][0] , positionsIn[j+1][1] , positionsIn[j+1][2]);
                glEnd();
            }
        }
    }
    // ----------------------------------------------------------------------------

    // TODO
    // a la fin on aura un maillage voxelisé genre
    // du coup on réapplique un HPPS aux sommets de ce maillage histoire de lisser notre résultat
    // on ft la meme chose pr différentes résolutions
}

void draw () {
    glPointSize(2); // for example...

    glColor3f(0.8,0.8,1);
    drawPointSet(positions , normals);

    //displayGrid(Vec3(0.0,0.0,0.0), 1.5, 32);
    //dualContouring(kdtree);

    glColor3f(1,0.5,0.5);
    drawPointSet(positions2 , normals2);
}








void display () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    draw ();
    glFlush ();
    glutSwapBuffers ();
}

void idle () {
    glutPostRedisplay ();
}

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }
        break;

    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if(polygonMode[0] != GL_FILL)
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        break;

    default:
        break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}





int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("tp point processing");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);


    {
        // Load a first pointset, and build a kd-tree:
        loadPN("pointsets/igea.pn" , positions , normals);
        //loadPN("pointsets/dino_subsampled_extreme.pn" , positions , normals);


        BasicANNkdTree kdtree;
        kdtree.build(positions);

        // -------------------------------------TP2 ------------------------------------------------------
        dualContouring(kdtree);
        // --------------------------------------------------------------------------------------------

		// -------------------------------------TP1 ------------------------------------------------------
        // Create a second pointset that is artificial, and project it on pointset1 using MLS techniques:
        //positions2.resize( 20000 );
        //normals2.resize(positions2.size());
        
        // nuage de points circulaire
        /*for( unsigned int pIt = 0 ; pIt < positions2.size() ; ++pIt ) {
            positions2[pIt] = Vec3(
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX),
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX),
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX)
                        );
            positions2[pIt].normalize();
            positions2[pIt] = 0.6 * positions2[pIt];
        }*/

        // nuage de point dans le cube
        /*for (unsigned int pIt = 0; pIt < positions2.size(); ++pIt) {
		    positions2[pIt] = Vec3(
		        -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX),  // X coordinate in [-2, 2]
		        -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX),  // Y coordinate in [-2, 2]
		        -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX)   // Z coordinate in [-2, 2]
		    );
		    // No need to normalize, as we want the points to be distributed within the cube boundaries.
		}*/

        // PROJECT USING MLS (HPSS and APSS):
        // But : projeter tous les points rouges sur la surface de l'objet 
        /*for(unsigned int i = 0; i < positions2.size(); i++){
        	HPSS(positions2[i], positions2[i], normals2[i], positions, normals, kdtree, 0, 1.0);
        }*/
        // --------------------------------------------------------------------------------------------
    }



    glutMainLoop ();
    return EXIT_SUCCESS;
}

