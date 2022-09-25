#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Eigen/Eigen"
#include <math.h>
#include <iostream>
#include "OBJ_Loader.h"

#define PI 3.141592653

using namespace std;

namespace Application {

	Eigen::Matrix<float, 4, 4> rodrigues_rotation(Eigen::Vector3f n, float angle)
	{
		float radian = angle / 180 * PI;
		float nx = n[0];
		float ny = n[1];
		float nz = n[2];

		Eigen::Matrix<float, 3, 3> M;
		M << 0, -nz, ny,
			nz, 0, -nx,
			-ny, nx, 0;
		Eigen::Matrix<float, 4, 4> R = Eigen::Matrix<float, 4, 4>::Identity();
		R.block<3, 3>(0, 0) =
			cos(radian) * Eigen::Matrix<float, 3, 3>::Identity() +
			(1 - cos(radian)) * n * n.transpose() +
			sin(radian) * M;
		return R;
	}

	Eigen::Matrix<float, 4, 4> get_view_matrix(
		Eigen::Vector3f e,
		Eigen::Vector3f g,
		Eigen::Vector3f t)
	{
		Eigen::Matrix<float, 4, 4> T;
		T << 1, 0, 0, -e[0],
			0, 1, 0, -e[1],
			0, 0, 1, -e[2],
			0, 0, 0, 1;

		auto g_cross = g.cross(t);
		Eigen::Matrix<float, 4, 4> R;
		R << g_cross[0], t[0], -g[0], 0,
			g_cross[1], t[1], -g[1], 0,
			g_cross[2], t[2], -g[2], 0,
			0, 0, 0, 1;
		return R.transpose() * T;
		//return T;
	}

	Eigen::Matrix4f get_perspective_matrix(float eye_fov, float aspect_ratio,
		float near, float far)
	{
		Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

		float t = -abs(near) * tan(eye_fov / 180 * PI / 2.0);
		float r = t * aspect_ratio;

		projection <<
			near / r, 0, 0, 0,
			0, near / t, 0, 0,
			0, 0, (near + far) / (near - far), -2 * near * far / (near - far),
			0, 0, 1, 0;
		return projection;
	}

	Eigen::Matrix<float, 4, 4> get_viewport_matrix(
		float width, float height, float near, float far)
	{
		Eigen::Matrix<float, 4, 4> M;
		M << width / 2, 0, 0, width / 2,
			0, -height / 2, 0, height / 2,
			0, 0, -(near - far) / 2.0, (near + far) / 2.0,
			0, 0, 0, 1;
		return M;
	}

	// 根据给定的x、y，计算这个点的统一表示，w1, w2, w3
	// P = w1 * A + w2 * B + w3 * C
	std::tuple<float, float, float> barycentric(float x, float y, float ax, float ay, float bx, float by, float cx, float cy)
	{
		float c1 = (x * (by - cy) + (cx - bx) * y + bx * cy - cx * by) / (ax * (by - cy) + (cx - bx) * ay + bx * cy - cx * by);
		float c2 = (x * (cy - ay) + (ax - cx) * y + cx * ay - ax * cy) / (bx * (cy - ay) + (ax - cx) * by + cx * ay - ax * cy);
		float c3 = (x * (ay - by) + (bx - ax) * y + ax * by - bx * ay) / (cx * (ay - by) + (bx - ax) * cy + ax * by - bx * ay);
		return {c1, c2, c3};
	}

	float cross(float x1, float y1, float x2, float y2) {
		return x1 * y2 - y1 * x2;
	}

	bool inside(
		float ax, float ay, float bx, float by, float cx, float cy,
		float px, float py
	) {
		// a, b, c
		// (b-a) x (p-a)
		// (c-b) x (p-b)
		// (a-c) x (p-c)

		float r0 = cross(bx - ax, by - ay, px - ax, py - ay);
		float r1 = cross(cx - bx, cy - by, px - bx, py - by);
		float r2 = cross(ax - cx, ay - cy, px - cx, py - cy);
		return r0 <= 0 && r1 <= 0 && r2 <= 0 || r0 >= 0 && r1 >= 0 && r2 >= 0;
	}

	class Object {
	public:
		Object& set_texture(const cv::Mat& image) {
			texture_ = image;
			return *this;
		}

		// 定义纹理uv坐标
		Object& set_texcoords(const vector<Eigen::Vector2f>& texcoords) {
			texcoords_ = texcoords;
			return *this;
		}

		// 定义顶点
		Object& set_vertexs(const vector<Eigen::Vector4f>& vertexs) {
			vertexs_ = vertexs;
			return *this;
		}

		// 定义法向量
		Object& set_normals(const vector<Eigen::Vector3f>& normals) {
			normals_ = normals;
			return *this;
		}

		// 定义三角形的索引
		Object& set_triangles(const vector<Eigen::Vector3i>& triangles) {
			triangles_ = triangles;
			return *this;
		}

		Object& set_model_matrix(const Eigen::Matrix4f& matrix) {
			model_matrix_ = matrix;
			return *this;
		}

	public:
		cv::Mat texture_;
		vector<Eigen::Vector2f> texcoords_;
		vector<Eigen::Vector4f> vertexs_;
		vector<Eigen::Vector3f> normals_;
		vector<Eigen::Vector3i> triangles_;
		Eigen::Matrix4f model_matrix_;
	};

	class ImageObject : public Object {
	public:
		ImageObject(const cv::Mat& texture, const vector<Eigen::Vector4f>& corners) {
			// corners format are:  a b c d
			//  d     c
			//  a     b
			this->set_vertexs(corners);
			this->set_texture(texture);
			this->set_triangles({Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(2, 3, 0)});
			this->set_texcoords({
				Eigen::Vector2f(0, 1), Eigen::Vector2f(1, 1),
				Eigen::Vector2f(1, 0), Eigen::Vector2f(0, 0)
			});
		}
	};

	std::tuple<Eigen::Vector3f, float> project(const Eigen::Matrix4f& mvp, const Eigen::Vector4f& pos) {
		auto output_v4 = (mvp * pos).eval();
		return std::make_tuple((output_v4 / output_v4[3]).head<3>(), output_v4[3]);
	}

	void draw_object_msaa4x(cv::Mat& scene, vector<float>& zbuffer, vector<Eigen::Vector3f>& sample_buffer, 
		Eigen::Matrix4f vp_matrix, Eigen::Matrix4f view_matrix, Eigen::Vector3f eye_pos, const Object& object
	) {
		
		vector<Eigen::Vector3f> l1{{15, 15, 15}, {500, 500, 500}};
		vector<Eigen::Vector3f> l3{{-20, 20, 0}, {500, 500, 500}};
		vector<Eigen::Vector3f> l4{{0, 0, -50}, {500, 500, 500}};
		std::vector<vector<Eigen::Vector3f>> lights = {l1, l3, l4};
		Eigen::Vector3f amb_light_intensity{10, 10, 10};

		auto mvp = (vp_matrix * object.model_matrix_).eval();
		auto mv = (view_matrix * object.model_matrix_).eval();
		for (int i = 0; i < object.triangles_.size(); ++i) {
			auto& tri = object.triangles_[i];
			int ia = tri[0];
			int ib = tri[1];
			int ic = tri[2];

			Eigen::Vector3f va, vb, vc;
			float wa, wb, wc;

			tie(va, wa) = project(mvp, object.vertexs_[ia]);
			tie(vb, wb) = project(mvp, object.vertexs_[ib]);
			tie(vc, wc) = project(mvp, object.vertexs_[ic]);

			Eigen::Vector2f texcoord_a = object.texcoords_[ia];
			Eigen::Vector2f texcoord_b = object.texcoords_[ib];
			Eigen::Vector2f texcoord_c = object.texcoords_[ic];
			Eigen::Vector3f norm_a = object.normals_[ia];
			Eigen::Vector3f norm_b = object.normals_[ib];
			Eigen::Vector3f norm_c = object.normals_[ic];

			vector<Eigen::Vector3f> view_position{
				(mv * object.vertexs_[ia]).head<3>(),
				(mv * object.vertexs_[ib]).head<3>(),
				(mv * object.vertexs_[ic]).head<3>()
			};

			int bleft   = min((float)scene.cols - 1, max(0.0f, min(min(va.x(), vb.x()), vc.x())));
			int btop    = min((float)scene.rows - 1, max(0.0f, min(min(va.y(), vb.y()), vc.y())));
			int bright  = min((float)scene.cols - 1, max(0.0f, ceil(max(max(va.x(), vb.x()), vc.x()))));
			int bbottom = min((float)scene.rows - 1, max(0.0f, ceil(max(max(va.y(), vb.y()), vc.y()))));

			for (int y = btop; y <= bbottom; ++y) {
				for (int x = bleft; x <= bright; ++x) {
					float x_offset[] = {0.25, 0.75, 0.75, 0.25};
					float y_offset[] = {0.25, 0.25, 0.75, 0.75};
					int hit_pixel_count = 0;
					const int n_sample_points = 4;
					int pixel_index = y * scene.cols + x;
					float alpha, beta, gamma;
					std::tie(alpha, beta, gamma) = barycentric(x+0.5f, y+0.5f, va.x(), va.y(), vb.x(), vb.y(), vc.x(), vc.y());

					for (int isample = 0; isample < n_sample_points; ++isample) {
						float fx = x + x_offset[isample];
						float fy = y + y_offset[isample];
						if (inside(
							va.x(), va.y(), vb.x(), vb.y(), vc.x(), vc.y(), fx, fy
						)) {
							float ww = alpha / wa + beta / wb + gamma / wc;
							float z_interpolated = (va.z() * alpha / wa + vb.z() * beta / wb + vc.z() * gamma / wc) / ww;
							int sample_index = pixel_index * n_sample_points + isample;

							if (z_interpolated < zbuffer[sample_index]) {

								auto uv = (texcoord_a * alpha / wa + texcoord_b * beta / wb + texcoord_c * gamma / wc) / ww;
								int u = min(object.texture_.cols - 1.0f, max(0.0f, uv.x() * object.texture_.cols + 0.5f));
								int v = min(object.texture_.rows - 1.0f, max(0.0f, (1- uv.y()) * object.texture_.rows + 0.5f));

								hit_pixel_count += 1;

								Eigen::Vector3f color;
								if (object.texture_.channels() == 4) {
									auto& obj_pixel = object.texture_.at<cv::Vec4b>(v, u);
									auto& scene_pixel = scene.at<cv::Vec3b>(y, x);
									float transparent = obj_pixel[3] / 255.0f;
									color =
										Eigen::Vector3f(scene_pixel[0], scene_pixel[1], scene_pixel[2]) * (1 - transparent) + 
										Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]) * transparent;
								}
								else {
									auto& obj_pixel = object.texture_.at<cv::Vec3b>(v, u);
									color = Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]);
								}

								Eigen::Vector3f interpolated_normal = alpha * norm_a + beta * norm_b + gamma * norm_c;
								Eigen::Vector3f interpolated_shadingcoords = alpha * view_position[0] + beta * view_position[1] + gamma * view_position[2];
								Eigen::Vector3f ka(0.005, 0.005, 0.005);  // 环境反射
								Eigen::Vector3f kd = color / 255.f;   // 漫反射
								Eigen::Vector3f ks(0.1, 0.1, 0.1);  // 镜面反射

								float p = 150;
								Eigen::Vector3f point = interpolated_shadingcoords;
								Eigen::Vector3f normal = interpolated_normal;
								Eigen::Vector3f result_color = {0, 0, 0};

								for (auto &light : lights)
								{
									Eigen::Vector3f l = (light[0] - point).normalized();
									Eigen::Vector3f v = (eye_pos - point).normalized();
									Eigen::Vector3f h = (l + v).normalized();
									float attenuation = 1.0f / (light[0] - point).squaredNorm();
									Eigen::Vector3f diffuse = kd.cwiseProduct(light[1]) * attenuation * std::max(normal.dot(l), 0.f);
									Eigen::Vector3f specular = ks.cwiseProduct(light[1]) * attenuation * pow(std::max(normal.dot(h), 0.f), p);
									result_color += diffuse + specular;
								}
								Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
								result_color += ambient;
								result_color *= 255;
								sample_buffer[sample_index] = result_color;
								zbuffer[sample_index] = z_interpolated;
							}
						}
					}

					if (hit_pixel_count > 0) {
						Eigen::Vector3f avgrage_pixel(0, 0, 0);
						int sample_index = pixel_index * n_sample_points;

						for (int i = 0; i < n_sample_points; i++)
							avgrage_pixel += sample_buffer[sample_index + i];

						avgrage_pixel /= n_sample_points;
						scene.at<cv::Vec3b>(y, x) = cv::Vec3b(avgrage_pixel.x(), avgrage_pixel.y(), avgrage_pixel.z());
					}
				}
			}
		}
	}

	void draw_object_naive(cv::Mat& scene, vector<float>& zbuffer, vector<Eigen::Vector3f>& sample_buffer, 
		Eigen::Matrix4f vp_matrix, Eigen::Matrix4f view_matrix, Eigen::Vector3f eye_pos, const Object& object
	) {
		vector<Eigen::Vector3f> l1{{15, 15, 15}, {500, 500, 500}};
		vector<Eigen::Vector3f> l3{{-20, 20, 0}, {500, 500, 500}};
		vector<Eigen::Vector3f> l4{{0, 0, -50}, {500, 500, 500}};
		std::vector<vector<Eigen::Vector3f>> lights = {l1, l3, l4};
		Eigen::Vector3f amb_light_intensity{10, 10, 10};

		auto mvp = (vp_matrix * object.model_matrix_).eval();
		auto mv = (view_matrix * object.model_matrix_).eval();
		for (int i = 0; i < object.triangles_.size(); ++i) {
			auto& tri = object.triangles_[i];
			int ia = tri[0];
			int ib = tri[1];
			int ic = tri[2];

			Eigen::Vector3f va, vb, vc;
			float wa, wb, wc;

			tie(va, wa) = project(mvp, object.vertexs_[ia]);
			tie(vb, wb) = project(mvp, object.vertexs_[ib]);
			tie(vc, wc) = project(mvp, object.vertexs_[ic]);

			Eigen::Vector2f texcoord_a = object.texcoords_[ia];
			Eigen::Vector2f texcoord_b = object.texcoords_[ib];
			Eigen::Vector2f texcoord_c = object.texcoords_[ic];
			Eigen::Vector3f norm_a = object.normals_[ia];
			Eigen::Vector3f norm_b = object.normals_[ib];
			Eigen::Vector3f norm_c = object.normals_[ic];

			vector<Eigen::Vector3f> view_position{
				(mv * object.vertexs_[ia]).head<3>(),
				(mv * object.vertexs_[ib]).head<3>(),
				(mv * object.vertexs_[ic]).head<3>()
			};

			int bleft = min((float)scene.cols - 1, max(0.0f, min(min(va.x(), vb.x()), vc.x())));
			int btop = min((float)scene.rows - 1, max(0.0f, min(min(va.y(), vb.y()), vc.y())));
			int bright = min((float)scene.cols - 1, max(0.0f, ceil(max(max(va.x(), vb.x()), vc.x()))));
			int bbottom = min((float)scene.rows - 1, max(0.0f, ceil(max(max(va.y(), vb.y()), vc.y()))));

			for (int y = btop; y <= bbottom; ++y) {
				for (int x = bleft; x <= bright; ++x) {
					int sample_index = y * scene.cols + x;
					float fx = x + 0.5;
					float fy = y + 0.5;
					if (inside(
						va.x(), va.y(), vb.x(), vb.y(), vc.x(), vc.y(), fx, fy
					)) {
						float alpha, beta, gamma;
						std::tie(alpha, beta, gamma) = barycentric(fx, fy, va.x(), va.y(), vb.x(), vb.y(), vc.x(), vc.y());

						float ww = alpha / wa + beta / wb + gamma / wc;
						float z_interpolated = (va.z() * alpha / wa + vb.z() * beta / wb + vc.z() * gamma / wc) / ww;

						if (z_interpolated < zbuffer[sample_index]) {

							auto uv = (texcoord_a * alpha / wa + texcoord_b * beta / wb + texcoord_c * gamma / wc) / ww;
							int u = min(object.texture_.cols - 1.0f, max(0.0f, uv.x() * object.texture_.cols + 0.5f));
							int v = min(object.texture_.rows - 1.0f, max(0.0f, (1 - uv.y()) * object.texture_.rows + 0.5f));

							Eigen::Vector3f color;
							if (object.texture_.channels() == 4) {
								auto& obj_pixel = object.texture_.at<cv::Vec4b>(v, u);
								auto& scene_pixel = scene.at<cv::Vec3b>(y, x);
								float transparent = obj_pixel[3] / 255.0f;
								color =
									Eigen::Vector3f(scene_pixel[0], scene_pixel[1], scene_pixel[2]) * (1 - transparent) +
									Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]) * transparent;
							}
							else {
								auto& obj_pixel = object.texture_.at<cv::Vec3b>(v, u);
								color = Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]);
							}
							zbuffer[sample_index] = z_interpolated;

							Eigen::Vector3f interpolated_normal = alpha * norm_a + beta * norm_b + gamma * norm_c;
							Eigen::Vector3f interpolated_shadingcoords = alpha * view_position[0] + beta * view_position[1] + gamma * view_position[2];
							Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);  // 环境反射
							Eigen::Vector3f kd = color / 255.f;   // 漫反射
							Eigen::Vector3f ks = Eigen::Vector3f(0.1, 0.1, 0.1);  // 镜面反射

							float p = 150;
							Eigen::Vector3f point = interpolated_shadingcoords;
							Eigen::Vector3f normal = interpolated_normal;
							Eigen::Vector3f result_color = {0, 0, 0};

							for (auto &light : lights)
							{
								Eigen::Vector3f l = (light[0] - point).normalized();
								Eigen::Vector3f v = (eye_pos - point).normalized();
								Eigen::Vector3f h = (l + v).normalized();
								float attenuation = 1.0f / (light[0] - point).squaredNorm();
								Eigen::Vector3f diffuse = kd.cwiseProduct(light[1]) * attenuation * std::max(normal.dot(l), 0.f);
								Eigen::Vector3f specular = ks.cwiseProduct(light[1]) * attenuation * pow(std::max(normal.dot(h), 0.f), p);
								result_color += diffuse + specular;
							}
							Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
							result_color += ambient;
							result_color *= 255;
							scene.at<cv::Vec3b>(y, x) = cv::Vec3b(result_color.x(), result_color.y(), result_color.z()); ;
						}
					}
				}
			}
		}
	}

	vector<Eigen::Vector3f> make_msaa4x_sample_buffer(const cv::Mat& scene) {

		cv::Mat scene4x;
		vector<Eigen::Vector3f> sample_buffer(scene.size().area() * 4);
		cv::resize(scene, scene4x, cv::Size(), 2, 2, cv::INTER_AREA);

		for (int i = 0; i < scene4x.rows; ++i) {
			for (int j = 0; j < scene4x.cols; ++j) {
				int ix = j / 2;
				int iy = i / 2;
				int is = (j % 2) + (i % 2) * 2;
				cv::Vec3b& pixel = scene4x.at<cv::Vec3b>(i, j);
				sample_buffer[(iy * scene.cols + ix) * 4 + is] = Eigen::Vector3f(pixel[0], pixel[1], pixel[2]);
			}
		}
		return sample_buffer;
	}

	int main()
	{
		objl::Loader loader;
		bool loadout = loader.LoadFile("spot_triangulated_good.obj");
		auto texture = cv::imread("spot_texture.png");
		auto background = cv::imread("background.png");
		Object obj;
		obj.set_texture(texture);

		for (auto mesh : loader.LoadedMeshes){
			for (int i = 0; i < mesh.Vertices.size(); i += 3){
				Eigen::Vector3i triangle(obj.vertexs_.size()+0, obj.vertexs_.size()+1, obj.vertexs_.size()+2);
				for (int j = 0; j < 3; j++){
					auto& veritices = mesh.Vertices[i + j];
					obj.vertexs_.emplace_back(veritices.Position.X, veritices.Position.Y, veritices.Position.Z, 1.0f);
					obj.normals_.emplace_back(veritices.Normal.X, veritices.Normal.Y, veritices.Normal.Z);
					obj.texcoords_.emplace_back(veritices.TextureCoordinate.X, veritices.TextureCoordinate.Y);
				}
				obj.triangles_.emplace_back(triangle);
			}
		}

		float ax = 0;
		float ay = 135;
		float az = 0;
		float camera_z = 3.5;
		float scene_width = 800;
		float scene_height = 800;
		float near = 0.1;
		float far = 50;
		float fov = 45;
		cv::Mat scene(scene_height, scene_width, CV_8UC3);
		cv::resize(background, scene, cv::Size(scene_width, scene_height));

		vector<float> zbuffer(scene_height * scene_width * 4);
		vector<Eigen::Vector3f> sample_buffer_background = make_msaa4x_sample_buffer(scene);
		vector<Eigen::Vector3f> sample_buffer = sample_buffer_background;

		auto e = Eigen::Vector3f(0, 0, camera_z);
		auto g = Eigen::Vector3f(0, 0, -1);
		auto t = Eigen::Vector3f(0, 1, 0);
		auto viewport_matrix = get_viewport_matrix(scene_width, scene_height, near, far);
		auto perspective_matrix = get_perspective_matrix(fov, 1, near, far);
		auto view_matrix = get_view_matrix(e, g, t);
		auto ortho_matrix = viewport_matrix * perspective_matrix * view_matrix;

		auto model_matrix = (
			rodrigues_rotation(Eigen::Vector3f(0, 0, 1), az) *
			rodrigues_rotation(Eigen::Vector3f(1, 0, 0), ax) *
			rodrigues_rotation(Eigen::Vector3f(0, 1, 0), ay)).eval();

		cv::resize(background, scene, cv::Size(scene_width, scene_height));
		std::fill(zbuffer.begin(), zbuffer.end(), std::numeric_limits<float>::max());
		sample_buffer = sample_buffer_background;

		obj.set_model_matrix(model_matrix);
		//draw_object_naive(scene, zbuffer, sample_buffer, ortho_matrix, view_matrix, e, obj);
		draw_object_msaa4x(scene, zbuffer, sample_buffer, ortho_matrix, view_matrix, e, obj);
		cv::imwrite("scene.jpg", scene);
		return 0;
	}
};

int main() {
	return Application::main();
}