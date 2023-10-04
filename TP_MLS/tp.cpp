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


// generate the eight corners of a box described by its extreme corners (max_corner and min_corner)
std::vector<Vec3> generateCorners(Vec3 max_corner, Vec3 min_corner){
    Vec3 TLF_corner(min_corner[0], max_corner[1], min_corner[2]); // Top Left Front
    Vec3 TLB_corner(min_corner[0], max_corner[1], max_corner[2]); // Top Left Back
    Vec3 TRF_corner(max_corner[0], max_corner[1], min_corner[2]); // Top Right Back
    Vec3 DLB_corner(min_corner[0], min_corner[1], max_corner[2]); // Down Left Front
    Vec3 DRF_corner(max_corner[0], min_corner[1], min_corner[2]); // Down Right Front
    Vec3 DRB_corner(max_corner[0], min_corner[1], max_corner[2]); // Down Right Back

    std::vector<Vec3> corners{max_corner, TRF_corner, TLF_corner, TLB_corner, DLB_corner, min_corner, DRF_corner, DRB_corner};
    return corners;
}

// returns the center point of the bounding box described by max_corner and min_corner
Vec3 getCenter(Vec3 max_corner, Vec3 min_corner){
    double center_x = min_corner[0] + (max_corner[0] - min_corner[0])/2;
    double center_y = min_corner[1] + (max_corner[1] - min_corner[1])/2;
    double center_z = min_corner[2] + (max_corner[2] - min_corner[2])/2;
    Vec3 center(center_x, center_y, center_z);
    return center;
}

// displays a box giving its center point and size (+ color)
void displayVoxel(Vec3 center, double length, Vec3 min_corner, Vec3 max_corner, std::string color){
    // make voxel
    std::vector<Vec3> voxelCorners = generateCorners(max_corner, min_corner);

    // sketch voxel
    if(color.compare("yellow") == 0){
        glColor3f(0.8F, 1.0F, 0.0F);
    }else if(color.compare("blue") == 0){
        glColor3f(0.0F, 0.0F, 1.0F);
    }else if(color.compare("red") == 0){
        glColor3f(1.0F, 0.0F, 0.0F);
    }else if(color.compare("green") == 0){
        glColor3f(0.0F, 1.0F, 0.0F);
    }else if (color.compare("purple") == 0){
        glColor3f(0.5F, 0.0F, 0.5F);
    }
    glBegin(GL_LINE_STRIP);
    for(unsigned int i = 0 ; i < voxelCorners.size(); i++) {
        glVertex3f( voxelCorners[i][0] , voxelCorners[i][1] , voxelCorners[i][2] );
    }
    glVertex3f(voxelCorners[0][0], voxelCorners[0][1], voxelCorners[0][2]);
    glVertex3f(voxelCorners[3][0], voxelCorners[3][1], voxelCorners[3][2]);
    glVertex3f(voxelCorners[4][0], voxelCorners[4][1], voxelCorners[4][2]);
    glVertex3f(voxelCorners[7][0], voxelCorners[7][1], voxelCorners[7][2]);
    glVertex3f(voxelCorners[6][0], voxelCorners[6][1], voxelCorners[6][2]);
    glVertex3f(voxelCorners[1][0], voxelCorners[1][1], voxelCorners[1][2]);
    glVertex3f(voxelCorners[2][0], voxelCorners[2][1], voxelCorners[2][2]);
    glVertex3f(voxelCorners[5][0], voxelCorners[5][1], voxelCorners[5][2]);

    glEnd();
}

// displays the volume of a sphere given its center, radius and resolution
void displayGrid(Vec3 center, float length, int resolution){

	Vec3 min_corner(center[0]-(length/2), center[1]-(length/2), center[2]-(length/2));
    Vec3 max_corner(center[0]+(length/2), center[1]+(length/2), center[2]+(length/2));

    displayVoxel(center, length, min_corner, max_corner, "yellow");

    double voxel_length;
    double sq_dim = max_corner[0] - min_corner[0];
    voxel_length = sq_dim/(double)resolution;

    Vec3 current_min_corner = min_corner;
    Vec3 current_max_corner;
    for(int i = 0; i < resolution; i++){ // x-axis
        for(int j = 0; j < resolution; j++){ // y-axis
            for(int k = 0; k < resolution; k++){ // z-axis
                // update current_min_corner and current_max_corner
                current_min_corner[0] = min_corner[0] + i*voxel_length;
                current_min_corner[1] = min_corner[1] + j*voxel_length;
                current_min_corner[2] = min_corner[2] + k*voxel_length;

                current_max_corner = current_min_corner + Vec3(voxel_length, voxel_length, voxel_length);

                // generate the corners of the current voxel
                std::vector<Vec3> corners = generateCorners(current_max_corner, current_min_corner);

                Vec3 voxel_center = getCenter(current_max_corner, current_min_corner);
                displayVoxel(voxel_center, voxel_length, current_min_corner, current_max_corner, "blue");
            }
        }
    }
}


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

void draw () {
    glPointSize(2); // for example...

    glColor3f(0.8,0.8,1);
    drawPointSet(positions , normals);

    displayGrid(Vec3(0.0,0.0,0.0), 1.5, 2);

    //glColor3f(1,0.5,0.5);
    //drawPointSet(positions2 , normals2);
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
	unsigned int nbIterations = 10, unsigned int k = 20){

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

        // Create a second pointset that is artificial, and project it on pointset1 using MLS techniques:
        positions2.resize( 20000 );
        normals2.resize(positions2.size());
        
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
        for (unsigned int pIt = 0; pIt < positions2.size(); ++pIt) {
		    positions2[pIt] = Vec3(
		        -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX),  // X coordinate in [-2, 2]
		        -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX),  // Y coordinate in [-2, 2]
		        -0.6 + 1.2 * (double)(rand()) / (double)(RAND_MAX)   // Z coordinate in [-2, 2]
		    );
		    // No need to normalize, as we want the points to be distributed within the cube boundaries.
		}

        // PROJECT USING MLS (HPSS and APSS):
        // But : projeter tous les points rouges sur la surface de l'objet 
        /*for(unsigned int i = 0; i < positions2.size(); i++){
        	HPSS(positions2[i], positions2[i], normals2[i], positions, normals, kdtree, 0, 1.0);
        }*/
    }



    glutMainLoop ();
    return EXIT_SUCCESS;
}

