#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Eigen/Eigen"
#include <math.h>
#include <iostream>

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
		Object& set_vertexs(const vector<Eigen::Vector3f>& vertexs) {
			vertexs_ = vertexs;
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
		vector<Eigen::Vector3f> vertexs_;
		vector<Eigen::Vector3i> triangles_;
		Eigen::Matrix4f model_matrix_;
	};

	class ImageObject : public Object {
	public:
		ImageObject(const cv::Mat& texture, const vector<Eigen::Vector3f>& corners) {
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

	std::tuple<Eigen::Vector3f, float> project(const Eigen::Matrix4f& mvp, const Eigen::Vector3f& pos) {
		auto output_v4 = (mvp * Eigen::Vector4f(pos[0], pos[1], pos[2], 1)).eval();
		return std::make_tuple((output_v4 / output_v4[3]).head<3>(), output_v4[3]);
	}

	void draw_object_msaa4x(cv::Mat& scene, vector<float>& zbuffer, vector<Eigen::Vector3f>& sample_buffer, Eigen::Matrix4f vp_matrix, const Object& object) {
		
		auto mvp = vp_matrix * object.model_matrix_;
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

			int bleft   = min((float)scene.cols - 1, max(0.0f, min(min(va.x(), vb.x()), vc.x())));
			int btop    = min((float)scene.rows - 1, max(0.0f, min(min(va.y(), vb.y()), vc.y())));
			int bright  = min((float)scene.cols - 1, max(0.0f, ceil(max(max(va.x(), vb.x()), vc.x()))));
			int bbottom = min((float)scene.rows - 1, max(0.0f, ceil(max(max(va.y(), vb.y()), vc.y()))));

			for (int y = btop; y <= bbottom; ++y) {
				for (int x = bleft; x <= bright; ++x) {
					float x_offset[] = {0.25, 0.75, 0.25, 0.75};
					float y_offset[] = {0.25, 0.25, 0.75, 0.75};
					int hit_pixel_count = 0;
					const int n_sample_points = 4;
					int pixel_index = y * scene.cols + x;

					for (int isample = 0; isample < n_sample_points; ++isample) {
						float fx = x + x_offset[isample];
						float fy = y + y_offset[isample];
						if (inside(
							va.x(), va.y(), vb.x(), vb.y(), vc.x(), vc.y(), fx, fy
						)) {
							float alpha, beta, gamma;
							std::tie(alpha, beta, gamma) = barycentric(fx, fy, va.x(), va.y(), vb.x(), vb.y(), vc.x(), vc.y());

							float ww = alpha / wa + beta / wb + gamma / wc;
							float z_interpolated = (va.z() * alpha / wa + vb.z() * beta / wb + vc.z() * gamma / wc) / ww;
							int sample_index = pixel_index * n_sample_points + isample;

							if (z_interpolated < zbuffer[sample_index]) {

								auto uv = (texcoord_a * alpha / wa + texcoord_b * beta / wb + texcoord_c * gamma / wc) / ww;
								int u = min(object.texture_.cols - 1.0f, max(0.0f, uv.x() * object.texture_.cols + 0.5f));
								int v = min(object.texture_.rows - 1.0f, max(0.0f, uv.y() * object.texture_.rows + 0.5f));

								hit_pixel_count += 1;
								if (object.texture_.channels() == 4) {
									auto& obj_pixel = object.texture_.at<cv::Vec4b>(v, u);
									auto& scene_pixel = scene.at<cv::Vec3b>(y, x);
									float transparent = obj_pixel[3] / 255.0f;
									sample_buffer[sample_index] = 
										Eigen::Vector3f(scene_pixel[0], scene_pixel[1], scene_pixel[2]) * (1 - transparent) + 
										Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]) * transparent;
								}
								else {
									auto& obj_pixel = object.texture_.at<cv::Vec3b>(v, u);
									sample_buffer[sample_index] = Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]);
								}
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

	void draw_object_naive(cv::Mat& scene, vector<float>& zbuffer, vector<Eigen::Vector3f>& sample_buffer, Eigen::Matrix4f vp_matrix, const Object& object) {

		auto mvp = vp_matrix * object.model_matrix_;
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
							int v = min(object.texture_.rows - 1.0f, max(0.0f, uv.y() * object.texture_.rows + 0.5f));

							if (object.texture_.channels() == 4) {
								auto& obj_pixel = object.texture_.at<cv::Vec4b>(v, u);
								auto& scene_pixel = scene.at<cv::Vec3b>(y, x);
								float transparent = obj_pixel[3] / 255.0f;
								Eigen::Vector3f pixel = 
									Eigen::Vector3f(scene_pixel[0], scene_pixel[1], scene_pixel[2]) * (1 - transparent) +
									Eigen::Vector3f(obj_pixel[0], obj_pixel[1], obj_pixel[2]) * transparent;
								scene_pixel = cv::Vec3b(pixel[0], pixel[1], pixel[2]);
							}
							else {
								auto& obj_pixel = object.texture_.at<cv::Vec3b>(v, u);
								scene.at<cv::Vec3b>(y, x) = cv::Vec3b(obj_pixel[0], obj_pixel[1], obj_pixel[2]);
							}
							zbuffer[sample_index] = z_interpolated;
						}
					}
				}
			}
		}
	}

	cv::Mat add_foot_gradual_channel(const cv::Mat& image, float transparent_y_begin=0.5) {

		cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
		for (int i = mask.rows * transparent_y_begin; i < mask.rows; ++i) {
			float alpha = (i - mask.rows * transparent_y_begin) / (mask.rows * (1 - transparent_y_begin));
			alpha = alpha * alpha * 0.5;
			mask.row(i).setTo(alpha * 255);
		}

		cv::Mat channels[4];
		cv::split(image, channels);
		channels[3] = mask;

		cv::Mat output;
		cv::merge(channels, 4, output);
		return output;
	}

	cv::Mat make_border(const cv::Mat& image, int margin=5, cv::Scalar color=cv::Scalar::all(255)) {
		cv::Mat output;
		cv::copyMakeBorder(image, output, margin, margin, margin, margin, cv::BORDER_CONSTANT, color);
		return output;
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
		auto background = cv::imread("background.png");
		const char* grils[] = {
			"mn1.png", "mn2.png", "mn3.png", "mn4.png", "mn5.png"
		};

		int nimage = sizeof(grils) / sizeof(grils[0]);
		vector<ImageObject> image_objs;
		float angle = 0;
		float angle_step = 360 / (float)nimage;
		for (int i = 0; i < nimage; ++i) {
			auto image = make_border(cv::imread(grils[i]), 20);
			auto image_transparent = add_foot_gradual_channel(image);
			Eigen::Vector2f oa(1, -0.5);
			Eigen::Vector2f ob(1, +0.5);
			float radian = angle / 180.0f * PI;
			angle += angle_step;

			Eigen::Matrix2f m;
			m <<
				cos(radian), -sin(radian),
				-sin(radian), -cos(radian)
			;
			oa = m * oa;
			ob = m * ob;

			Eigen::Vector3f ba(oa.x(), -0.5, oa.y());
			Eigen::Vector3f bb(ob.x(), -0.5, ob.y());
			Eigen::Vector3f bc(ob.x(), 0.5, ob.y());
			Eigen::Vector3f bd(oa.x(), 0.5, oa.y());

			image_objs.push_back(ImageObject(
				image,
				{ba, bb, bc, bd}
			));

			bc.y() = -1.5;
			bd.y() = -1.5;
			image_objs.push_back(ImageObject(
				image_transparent,
				{ba, bb, bc, bd}
			));
		}

		float ax = 0;
		float ay = 0;
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
		int remaining_frames = 30 * 30;

		cv::VideoWriter vw;
		vw.open("output.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(scene_width, scene_height));

		while (remaining_frames--) {

			printf("remaining_frames = %d\n", remaining_frames);

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

			for (auto& obj : image_objs) {
				obj.set_model_matrix(model_matrix);
				draw_object_msaa4x(scene, zbuffer, sample_buffer, ortho_matrix, obj);
			}

			ay -= 0.5;
			vw.write(scene);
		};
		vw.release();
		return 0;
	}
};

int main() {
	return Application::main();
}