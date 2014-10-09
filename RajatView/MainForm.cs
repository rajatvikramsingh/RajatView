using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Windows.Forms;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using Emgu.CV;
using ZedGraph;

namespace RajatView
{
    public partial class MainForm : Form
    {
        public Image<Gray, byte> img = syntheticImage();
        Image<Gray, byte> b1, b2, b3, b4, b5, b6, b7, b8;
        double[,] h;

        public MainForm()
        {
            InitializeComponent();
            imageBox.Image = img;
        }

        public static Image<Gray, byte> syntheticImage()
        {
            Image<Gray, byte> imgnew = new Image<Gray, byte>(256, 256);
            Image<Gray, byte> imgnew1 = new Image<Gray, byte>(256, 256);
            Random r = new Random();
            Random r1 = new Random(128);
            imgnew1.SetValue(new Gray(0));
            int x = r.Next(imgnew.Width / 2);
            int y = r.Next(imgnew.Height);

            CvInvoke.cvRectangle(imgnew1.Ptr, new Point(x,x), new Point(y, y), new MCvScalar(r.Next(220)), 10, LINE_TYPE.EIGHT_CONNECTED, 0);
            CvInvoke.cvCircle(imgnew1.Ptr, new Point(r.Next(150), r.Next(255)), 20, new MCvScalar(r.Next(220)),5, LINE_TYPE.EIGHT_CONNECTED, 0);
            x = r1.Next(imgnew.Width);
            y = r1.Next(imgnew.Height);
            CvInvoke.cvRectangle(imgnew1.Ptr, new Point(x, x), new Point(y, y), new MCvScalar(r.Next(220)), 10, LINE_TYPE.EIGHT_CONNECTED, 0);
            CvInvoke.cvCircle(imgnew1.Ptr, new Point(r.Next(150), r.Next(255)), 20, new MCvScalar(r.Next(220)), 5, LINE_TYPE.EIGHT_CONNECTED, 0);
            x = r1.Next(imgnew.Width);
            y = r1.Next(imgnew.Height);
            CvInvoke.cvRectangle(imgnew1.Ptr, new Point(x, x), new Point(y, y), new MCvScalar(r.Next(220)), 10, LINE_TYPE.EIGHT_CONNECTED, 0);
            CvInvoke.cvCircle(imgnew1.Ptr, new Point(r.Next(150), r.Next(255)), 20, new MCvScalar(r.Next(220)), 5, LINE_TYPE.EIGHT_CONNECTED, 0);
            return imgnew1;
        }
        
        private void SAconvolve(int dim)
        {
            //Image<Gray, byte> imgnew = new Image<Gray, byte>(img.Height + 2 * (mask.Height - 1), img.Width + 2 * (mask.Width - 1));
            Image<Gray, byte> imgnew = new Image<Gray, byte>(img.Width, img.Height);
            double intensity = 0;
            int i,j=0;

            double[,] mask = genSAMask(img, dim);
            for (i = 0; i < imgnew.Height ; i++)
                for (j = 0; j < imgnew.Width ; j++)
                {
                    intensity = 0;
                    for(int k=0; k<dim; k++)
                        for (int l = 0; l < dim; l++)
                        {

                            if ((i + k - (dim / 2) >= 0) && (j + l - (dim / 2) >= 0) && (i + k - (dim / 2) < img.Height) && (j + l - (dim / 2) < img.Width))
                            {
                                intensity = intensity + (img[i + k - (dim / 2), j + l - (dim / 2)].Intensity * mask[k, l]);
                            }
                        }
                            imgnew[i, j] = new Gray(intensity / (dim * dim));
                        
                }
            showimage(imgnew);
        }

        private double[,] genSAMask(Image<Gray, byte> img, int dim)
        {
            double[,] mask = new double[dim, dim];
            for (int k = 0; k < dim; k++)
                for (int l = 0; l < dim; l++)
                    mask[k, l] = 1;
            return mask;
        }

        private void gaussconvolve(int dim)
        {
            //Image<Gray, byte> imgnew = new Image<Gray, byte>(img.Height + 2 * (mask.Height - 1), img.Width + 2 * (mask.Width - 1));
            Image<Gray, byte> imgnew = new Image<Gray, byte>(img.Height, img.Width);
            double[,] arr = new double[img.Height, img.Width];
            double intensity = 0;
            int i, j = 0;

            double[,] mask = gengaussMask(img, dim);
            for (i = 0; i < imgnew.Height; i++)
                for (j = 0; j < imgnew.Width; j++)
                {
                    intensity = 0;
                    for (int k = 0; k < dim; k++)
                        for (int l = 0; l < dim; l++)
                        {
                            //if ((i < mask.Height - 1 || i > img.Height + mask.Height - 1) && (j < mask.Width - 1 || j > img.Width + mask.Width - 1))
                            //    imgnew[i, j] = zero;
                            //else
                            //    imgnew[i, j] = img[i, j];

                            if ((i + k - 1 >= 0) && (j + l - 1 >= 0) && (i + k - 1 < img.Height) && (j + l - 1 < img.Width))
                            {
                                intensity = intensity + img[i + k - 1, j + l - 1].Intensity * mask[k, l];
                            }
                        }
                    arr[i, j] = intensity;
                    //imgnew[i, j] = new Gray(intensity);
                        
                }
            imgnew = scaleImage(arr, imgnew.Height, imgnew.Width);
            imageBox1.Image = imgnew;
        }


        private double[,] gengaussMask(Image<Gray, byte> img, int dim)
            {
                double[,] mask = new double[dim, dim];
                int x = dim - dim / 2, y = dim - dim / 2;
                int sigma = (int) numericUpDown4.Value;
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                    {
                        mask[i,j] = gauss(x, y, sigma);
                        x++;
                        y++;
                    }
                return mask;
            }


        private double gauss(int x, int y, int sigma)
        {
            double ans = 0;
            try
            {
                ans = Math.Exp(-(x * x + y * y) / (2 * sigma * sigma));
            }
            catch (System.DivideByZeroException exp)
            {
                Console.WriteLine(exp.Message);
            }
            return ans;
        }




        private Image<Gray,byte> convolve(Image<Gray, byte> img, double[,] mask, int dim)
        {
            //Image<Gray, byte> imgnew = new Image<Gray, byte>(img.Height + 2 * (mask.Height - 1), img.Width + 2 * (mask.Width - 1));
            Image<Gray, byte> imgnew = new Image<Gray, byte>(img.Height, img.Width);
            double intensity = 0;
            int i,j=0;
            for (i = 1; i < imgnew.Height-1; i++)
                for (j = 1; j < imgnew.Width-1; j++)
                {
                    intensity = 0;
                    for(int k=0; k<dim; k++)
                        for (int l = 0; l < dim; l++)
                        {
                            //if ((i < mask.Height - 1 || i > img.Height + mask.Height - 1) && (j < mask.Width - 1 || j > img.Width + mask.Width - 1))
                            //    imgnew[i, j] = zero;
                            //else
                            //    imgnew[i, j] = img[i, j];

                            if ((i + k - 1 >= 0) && (j + l - 1 >= 0) && (i + k - 1 < img.Height) && (j + l - 1 < img.Width))
                            {
                                intensity = intensity + img[i + k - 1, j + l - 1].Intensity * mask[k, l];
                            }
                        }
                    imgnew[i, j] = new Gray(intensity);
                }
            
            return(imgnew);
        }

        private void computeLaplacian()
        {
            double[,] mask = { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };
            Image<Gray, byte> imgnew = convolve(img, mask, 3);
            showimage(imgnew);
        }

        private Image<Gray, byte> computeretLaplacian()
        {
            double[,] mask = { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };
            Image<Gray, byte> imgnew = convolve(img, mask, 3);
            return imgnew;
        }

        private Image<Gray, byte> scaleImage(double[,] arr, int height, int width)
        {
            double min = arr[0,0];
            
            double[,] imgnew = new double[height,width];
            Image<Gray, byte> image = new Image<Gray, byte>(height, width);
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    if (arr[i, j] < min)
                        min = arr[i, j];
                }

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    imgnew[i, j] = arr[i, j] - min;
            double max = imgnew[0, 0];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    if (imgnew[i, j] > max)
                        max = imgnew[i, j];
                }
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    image[i, j] = new Gray((imgnew[i, j]/max)*255);
            return image;
        }



        private void computeSobel()
        {
            //Image<Gray, byte> img = new Image<Gray, byte>(openFileDialog1.FileName);
            double[,] maskx = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
            double[,] masky = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
            Image<Gray, byte> img1, img2, img3 = new Image<Gray, byte>(img.Height, img.Width);
            img1 = convolve(img, maskx, 3);
            img2 = convolve(img, masky, 3);
            for (int i=0; i<img.Height; i++)
                for (int j = 0; j < img.Width; j++)
                {
                    img3[i, j] = new Gray(Math.Sqrt((img1[i, j].Intensity * img1[i, j].Intensity) + (img2[i, j].Intensity * img2[i, j].Intensity)));
                }
            showimage(img3);
        }

        private void sharpenImage()
        {
            Image<Gray, byte> imgnew = computeretLaplacian();
            Image<Gray, byte> imgsharp = addImage(img, imgnew);
            showimage(imgsharp);
        }

        private Image<Gray, byte> addImage(Image<Gray, Byte> img1, Image<Gray, Byte> img2)
        {
            Image<Gray, Byte> img4 = new Image<Gray, byte>(img1.Width, img1.Height);
            for (int i = 0; i < img1.Height; i++)
            {
                for (int j = 0; j < img1.Width; j++)
                {
                    double z = 0;
                    z = img1[i, j].Intensity + img2[i, j].Intensity;

                    if (z > 255)
                        z = 255;
                    img4[i, j] = new Gray(z);

                }
            }
            //imageBox3.Image = img4;
            return img4;
        }

        public int[] dec2bin(int pix)
        {
            int k = 0;
            int remain;
            int[] temp = new int[8];

            do
            {
                remain = pix % 2;
                pix = pix / 2;
                temp[k++] = remain;
            } while (pix > 0);
            return temp;
        }

        private void FFT()
        {
            //Image<Gray, byte> img = new Image<Gray, byte>(openFileDialog1.FileName);
            double[,] arr = new double[img.Height, img.Width];
            Image<Gray, byte> fft_img = new Image<Gray, byte>(img.Height, img.Width);
            double p = 0;
            for (int k = 0; k < img.Height; k++)
            {
                for (int l = 0; l < img.Width; l++)
                {
                    /*p = calculate_fft(img[k, l].Intensity, k, l, img.Height, img.Width);
                    
                    arr[k, l] = p;*/
                    Complex total = Complex.Zero;
                    for (int i = 0; i < img.Width; i++)
                    {
                        for (int j = 0; j < img.Height; j++)
                        {
                            Complex c1 = Complex.Multiply(Complex.ImaginaryOne, (2 * Math.PI));
                            double c2 = (((double)(k * i)) / ((double)img.Width)) + (((double)(l * j)) / ((double)img.Height));
                            Complex power = Complex.Multiply(c1, c2);
                            Complex exp = Complex.Exp(power);
                            Complex answer;
                            if ((i + j) % 2 != 0)
                                answer = Complex.Divide((-img[i, j].Intensity), exp);
                            else
                                answer = Complex.Divide(img[i, j].Intensity, exp);
                            total = Complex.Add(total, answer);
                        }
                    }
                    total = Complex.Divide(total, (img.Width * img.Height));
                    double value = 1 + Math.Log10(total.Magnitude);
                    arr[k, l] = value;
                }
            }

            fft_img = scaleImage(arr, fft_img.Height, fft_img.Width);
            showimage(fft_img);
            
        }

        /*private double calculate_fft(double val, int k, int l, int height, int width)
        {
            double x = 0.0;
            Complex total = Complex.Zero;
            /*
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    
                    Complex exp = Complex.Multiply(Complex.Multiply(Complex.ImaginaryOne, (Math.PI * 2)), (((double)(i * k) / height) + ((double)(j * l) / width)));
                    Complex exp1 = Complex.Exp(exp);
                    Complex result = Complex.Zero;
                    if((i + j) % 2 == 0)
                        result = Complex.Divide(val, exp1);
                    else
                        result = Complex.Divide(-val, exp1);
                    total = Complex.Add(total, result);
                    */
                    //Complex c = new Complex(0, ((-2) * 3.14 * (i * k / height + j * l / width)));
                    //x = x + (val * Math.Exp(Math.Sqrt(Complex.Imag(c))));
/*
                    Complex sum = Complex.Zero;
                    for (int i = 0; i < width; i++)
                    {
                        for (int j = 0; j < height; j++)
                        {
                            Complex power1 = Complex.Multiply(Complex.ImaginaryOne, (2 * Math.PI));
                            double power2 = (((double)(k * i)) / ((double)width)) + (((double)(l * j)) / ((double)height));
                            Complex power = Complex.Multiply(power1, power2);
                            Complex exp = Complex.Exp(power);
                            Complex answer;
                            if ((i + j) % 2 != 0)
                                answer = Complex.Divide(-val, exp);
                            else
                                answer = Complex.Divide(val, exp);
                            sum = Complex.Add(sum, answer);
                        }
                    }
                    sum = Complex.Divide(sum, (width * height));
                    double value = 1 + Math.Log10(sum.Magnitude);
                    return value;
                }
            }
            total = Complex.Divide(total, height * width);
            double Pix_value = 1 + Math.Log10(total.Magnitude);

            return Pix_value;
        }*/

        private void inversFFT()
        {

            Image<Gray, byte> ifft_img = new Image<Gray, byte>(img.Height, img.Width);

            for (int i = 0; i < img.Height; i++)
            {
                for (int j = 0; j < img.Width; j++)
                {
                    ifft_img[i, j] = new Gray(calculate_ifft(img[i, j].Intensity, i, j, img.Height, img.Width));
                }
            }
            showimage(ifft_img);

        }

        private double calculate_ifft(double val, int k, int l, int height, int width)
        {
            double x = 0.0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    x = x + (val * Math.Exp(Math.Sqrt(-1) * (2) * 3.14 * (i * k / height + j * l / width)));
                }

            }
            return x;
        }

        private void log_toolStripButton1_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> logimg = new Image<Gray, byte>(img.Height, img.Width);
            Gray x;
            int c = (int) numericUpDown1.Value;
            for (int i = 0; i < img.Height; i++)
                for (int j = 0; j < img.Width; j++)
                {
                    x = new Gray(c * Math.Log(1 + img[i, j].Intensity));
                    logimg[i, j] = x;
                }
            showimage(logimg);
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            try
            {
                img = new Image<Gray, byte>(openFileDialog1.FileName);
                imageBox.Image = img;
            }
            catch (System.ArgumentException exp)
            {
                System.Console.WriteLine(exp.Message);
            }
        }

        private void showimage(Image<Gray, byte> img)
        {
            imageBox1.Image = img;
            img = this.img;
        }

        private void ilog_toolStripButton2_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> ilogimg = new Image<Gray, byte>(img.Height, img.Width);
            Gray x;
            int c = (int)numericUpDown1.Value;
            for (int i = 0; i < img.Height; i++)
                for (int j = 0; j < img.Width; j++)
                {
                    x = new Gray(Math.Exp(img[i, j].Intensity/c) - 1);
                    ilogimg[i, j] = x;
                }
            showimage(ilogimg);
        }

        private void neg_toolStripButton4_Click(object sender, EventArgs e)
        {
            Image<Gray, Byte> img4 = new Image<Gray, byte>(img.Width, img.Height);
            for (int i = 0; i < img.Height; i++)
            {
                for (int j = 0; j < img.Width; j++)
                {
                    img4[i, j] = new Gray(255 - img[i, j].Intensity);

                }
            }
            showimage(img4);
        }

        private void gamma_toolStripButton3_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> gammaimg = new Image<Gray, byte>(img.Height, img.Width);
            double gamma = (int)numericUpDown2.Value, c = (int)numericUpDown1.Value;
            Gray x;
            for (int i = 0; i < img.Height; i++)
                for (int j = 0; j < img.Width; j++)
                {
                    x = new Gray(c * Math.Pow(img[i, j].Intensity, gamma));
                    gammaimg[i, j] = x;
                }
            showimage(gammaimg);
        }

        private void bitplane_toolStripSplitButton1_ButtonClick(object sender, EventArgs e)
        {
            Image<Gray, byte> bit_plane = img;

            b1 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b2 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b3 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b4 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b5 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b6 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b7 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);
            b8 = new Image<Gray, byte>(bit_plane.Height, bit_plane.Width);


            Double pix;
            int[] s = new int[8];
            int[] bitplanes = new int[8];
            double db;
            //Gray val;
            string k, k1;
            for (int i = 0; i < bit_plane.Height; i++)
            {
                for (int j = 0; j < bit_plane.Width; j++)
                {
                    pix = bit_plane[i, j].Intensity;
                    s = dec2bin((int)pix);
                    k1 = s[0].ToString();
                    k = k1 += "0000000";
                    db = Convert.ToDouble(k);
                    b1[i, j] = new Gray(db);

                    k1 = s[1].ToString();
                    k = k1 += "0000000";
                    b2[i, j] = new Gray(Convert.ToDouble(k));

                    k1 = s[2].ToString();
                    k = k1 += "0000000";
                    b3[i, j] = new Gray(Convert.ToDouble(k));

                    k1 = s[3].ToString();
                    k = k1 += "0000000";
                    b4[i, j] = new Gray(Convert.ToDouble(k));

                    k1 = s[4].ToString();
                    k = k1 += "0000000";
                    b5[i, j] = new Gray(Convert.ToDouble(k));

                    k1 = s[5].ToString();
                    k = k1 += "0000000";
                    b6[i, j] = new Gray(Convert.ToDouble(k));

                    k1 = s[6].ToString();
                    k = k1 += "0000000";
                    b7[i, j] = new Gray(Convert.ToDouble(k));

                    k1 = s[7].ToString();
                    k = k1 += "0000000";
                    b8[i, j] = new Gray(Convert.ToDouble(k));


                }
            }
        }

        private void toolStripMenuItem2_Click(object sender, EventArgs e)
        {
            showimage(b1);
        }
        
        private void bitPlane8ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            showimage(b8);
        }

        private void bitPlane3ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            showimage(b3);
        }

        private void bitPlane4ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            showimage(b4);
        }

        private void bitPlane5ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            showimage(b5);
        }

        private void bitPlane6ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            showimage(b6);
        }

        private void bitPlane7ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            showimage(b7);
        }

        private void toolStripMenuItem3_Click(object sender, EventArgs e)
        {
            showimage(b2);
        }

        private void histogram_toolStripButton5_Click(object sender, EventArgs e)
        {
            GraphForm f = new GraphForm();
            f.setImage(img);
            f.Show();
        }

        private void SAFilter_toolStripButton6_Click(object sender, EventArgs e)
        {
            SAconvolve((int)numericUpDown3.Value);
        }

        private void gaussfilter_toolStripButton7_Click(object sender, EventArgs e)
        {
            gaussconvolve((int)numericUpDown3.Value);
        }

        private void sobel_toolStripButton8_Click(object sender, EventArgs e)
        {
            computeSobel();
        }

        private void laplacian_toolStripButton9_Click(object sender, EventArgs e)
        {
            computeLaplacian();
        }

        private void sharpen_toolStripButton12_Click(object sender, EventArgs e)
        {
            sharpenImage();
        }

        private void fft_toolStripButton10_Click(object sender, EventArgs e)
        {
            FFT();
        }

        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {
            saveFileDialog1.ShowDialog();
            try
            {
                img.Save(saveFileDialog1.FileName);
            }
            catch (System.Exception exp)
            {
                System.Console.WriteLine(exp.Message);
            }
        }

        private void ifft_toolStripButton11_Click(object sender, EventArgs e)
        {
            inversFFT();
        }

        private void dilateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> img1 = open(img);
            showimage(img1);
        }

        private Image<Gray, byte> open(Image<Gray, byte> img2)
        {
            Image<Gray, byte> img1 = img2.ThresholdBinary(new Gray(128), new Gray(255));
            img1._Dilate(1);
            img1._Erode(1);
            return img1;
        }

        private void erodeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> img1 = close(img);
            showimage(img1);
        }

        private Image<Gray, byte> close(Image<Gray, byte> img2)
        {
            Image<Gray, byte> img1 = img2.ThresholdBinary(new Gray(128), new Gray(255));
            img1._Erode(1);
            img1._Dilate(1);
            return img1;
        }



        private void skeletonizeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Emgu.CV.Image<Emgu.CV.Structure.Gray, byte> img1 = new Image<Emgu.CV.Structure.Gray, byte>(img.ToBitmap()).Resize(500, 500, true);
            Emgu.CV.Image<Gray, byte> gray = img1.Convert<Gray, byte>();//convert to grayscale
            Emgu.CV.Image<Gray, byte> binary = gray.ThresholdBinary(new Gray(75), new Gray(255));//perform binarization
            IntPtr dsti = Emgu.CV.CvInvoke.cvCreateImage(Emgu.CV.CvInvoke.cvGetSize(binary), Emgu.CV.CvEnum.IPL_DEPTH.IPL_DEPTH_32F, 1);
            Emgu.CV.CvInvoke.cvDistTransform(binary, dsti, Emgu.CV.CvEnum.DIST_TYPE.CV_DIST_L2, 5, null, IntPtr.Zero);
            CvInvoke.cvLaplace(dsti, dsti, 7);
            CvInvoke.cvThreshold(dsti, dsti, 3, 255, Emgu.CV.CvEnum.THRESH.CV_THRESH_BINARY);
            Image<Gray, byte> fu = new Image<Gray, byte>(binary.Width, binary.Height);
            CvInvoke.cvConvertScaleAbs(binary.Ptr, fu.Ptr, 2, 1.5);
            CvInvoke.cvNot(fu.Ptr, fu.Ptr);
            fu._Dilate(1);
            fu._Erode(1);
            showimage(fu);
        }

        private void connectedComponentsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> img1 = img.ThresholdBinary(new Gray(200), new Gray(255));
            Image<Gray, byte> img2 = new Image<Gray, byte>(img1.Width, img1.Height);
            Image<Gray, byte> img3 = new Image<Gray, byte>(img1.Width, img1.Height);
            Contour<Point> contours;
            Random r = new Random(0);

            contours = img1.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_CCOMP);
            int inm = 0;
            int jnm = 0;

            for (; contours != null; contours = contours.HNext)
            {

                MCvScalar color = new MCvScalar(r.Next(255));
                MCvScalar color1 = new MCvScalar(0);
                Contour<Point> cnt = contours;
                for (; cnt != null; cnt = cnt.VNext)
                {
                    jnm++;
                }
                jnm--;
                CvInvoke.cvDrawContours(img2.Ptr, contours.Ptr, color, color1, 2, -1, LINE_TYPE.EIGHT_CONNECTED, new Point(0, 0));
                inm++;
            }
            int[] minimg = new int[inm];
            int[] maximg = new int[inm];
            int[] meanimg = new int[inm];
            int count = 0;
            int no = 0;
            for (; contours != null; contours = contours.HNext)
            {
                minimg[count] = 255;
                for (int i = 0; i < img.Height; i++)
                {
                    for (int j = 0; j < img.Width; j++)
                    {

                        if (contours.InContour(new Point(i, j)) > 0)
                        {
                            if (minimg[count] > img[i, j].Intensity)
                                minimg[count] = (int)img[i, j].Intensity;
                            if (maximg[count] < img[i, j].Intensity)
                                maximg[count] = (int)img[i, j].Intensity;

                            meanimg[count] += (int)img[i, j].Intensity;
                            no++;

                        }
                    }
                    meanimg[count] = meanimg[count] / no;
                }
                count++;
            }

            int euler = inm - jnm;
            /*ImageViewer iv = new ImageViewer();
            iv.Image = img2;
            iv.ShowDialog();*/
            showimage(img2);

        }

        private Image<Gray, float> filter(Image<Gray, float> img)
        {
            Image<Gray, float> im_pad = new Image<Gray, float>(img.Width * 2, img.Height * 2);
            Image<Gray, float> dft = new Image<Gray, float>(img.Width * 2, img.Height * 2);
            im_pad = pad(img);
            CvInvoke.cvDFT(im_pad.Ptr, dft.Ptr, CV_DXT.CV_DXT_FORWARD, 0);
            return dft;

        }


        private Image<Gray, float> pad(Image<Gray, float> ig)
        {
            Image<Gray, float> im_new = new Image<Gray, float>(ig.Width * 2, ig.Height * 2);
            Image<Gray, float> img_return = new Image<Gray, float>(ig.Width * 2, ig.Height * 2);
            double b = 0;

            for (int i = 0; i < ig.Rows; i++)
            {
                for (int j = 0; j < ig.Cols; j++)
                {
                    b = ig[i, j].Intensity;
                    b = b * ((-1) ^ (i + j));
                    img_return[i, j] = new Gray(b);
                }

            }
            return img_return;

        }

        private void idealToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Image<Gray, float> im_pad = filter(img.Convert<Gray, float>());
            Image<Gray, float> retimage = new Image<Gray, float>(2 * img.Width, 2 * img.Height);
            h = retidealmask(150, 0);
            retimage = convolve(im_pad, h);
            imageBox1.Image = retimage;
        }

        private Image<Gray, float> convolve(Image<Gray, float> img, double[,] h)
        {
            Image<Gray, float> ans = new Image<Gray, float>(img.Width, img.Height);
            Image<Gray, float> retimg = new Image<Gray, float>(img.Width, img.Height);
            Image<Gray, float> retimg1 = new Image<Gray, float>(img.Width / 2, img.Height / 2);
            for (int k = 0; k < img.Rows; k++)
                for (int l = 0; l < img.Cols; l++)
                {
                    ans[k, l] = new Gray(img[k, l].Intensity * h[k, l]);
                }
            CvInvoke.cvDFT(ans.Ptr, retimg.Ptr, CV_DXT.CV_DXT_INV_SCALE, 0);
            for (int k = 0; k < img.Rows / 2; k++)
                for (int l = 0; l < img.Cols / 2; l++)
                {
                    retimg1[k, l] = retimg[k, l];
                }

            return retimg1;
        }

        private void butterworthToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Image<Gray, float> im_pad = filter(img.Convert<Gray, float>());
            Image<Gray, float> retimage = new Image<Gray, float>(2 * img.Width, 2 * img.Height);
            h = retbutterworthmask(1, 0);
            retimage = convolve(im_pad, h);
            imageBox1.Image = retimage;
        }

        private void gaussianToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Image<Gray, float> im_pad = filter(img.Convert<Gray, float>());
            Image<Gray, float> retimage = new Image<Gray, float>(2 * img.Width, 2 * img.Height);
            h = retgaussianmask(0);
            retimage = convolve(im_pad, h);
            imageBox1.Image = retimage;

        }

        private double[,] retidealmask(int radius, int mode)
        {
            h = new double[img.Rows * 2, img.Cols * 2];
            for (int i = 0; i < 2 * img.Rows; i++)
                for (int j = 0; j < 2 * img.Cols; j++)
                {
                    if (Math.Sqrt(Math.Pow((i - (img.Width)), 2) + Math.Pow(j - (img.Height), 2)) <= radius)
                        h[i, j] = 1;
                    else
                        h[i, j] = 0;
                }
            if (mode == 0)
            {
                for (int i = 0; i < 2 * img.Rows; i++)
                    for (int j = 0; j < 2 * img.Cols; j++)
                    {
                        h[i, j] = 1 - h[i, j];
                    }
            }
            return h;
        }

        private double[,] retgaussianmask(int mode)
        {
            h = new double[img.Rows * 2, img.Cols * 2];
            for (int i = 0; i < 2 * img.Rows; i++)
                for (int j = 0; j < 2 * img.Cols; j++)
                {
                    h[i, j] = Math.Exp(((-1) * (Math.Pow((i - (img.Width)), 2) + Math.Pow(j - (img.Height), 2))) / (2 * Math.Pow(15, 2)));
                }
            if (mode == 0)
            {
                for (int i = 0; i < 2 * img.Rows; i++)
                    for (int j = 0; j < 2 * img.Cols; j++)
                    {
                        h[i, j] = 1 - h[i, j];
                    }
            }
            return h;
        }

        private double[,] retbutterworthmask(int n, int mode)
        {
            h = new double[img.Rows * 2, img.Cols * 2];
            for (int i = 0; i < 2 * img.Rows; i++)
                for (int j = 0; j < 2 * img.Cols; j++)
                {
                    h[i, j] = 1 / (Math.Pow((Math.Sqrt(Math.Pow((i - (img.Width)), 2) + Math.Pow(j - (img.Height), 2)) / 30), 2 * n) + 1);
                }
            if (mode == 0)
            {
                for (int i = 0; i < 2 * img.Rows; i++)
                    for (int j = 0; j < 2 * img.Cols; j++)
                    {
                        h[i, j] = 1 - h[i, j];
                    }
            }
            return h;
        }

        private void idealToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            Image<Gray, float> im_pad = filter(img.Convert<Gray, float>());
            Image<Gray, float> retimage = new Image<Gray, float>(2 * img.Width, 2 * img.Height);
            h = retidealmask(200, 1);
            retimage = convolve(im_pad, h);
            imageBox1.Image = retimage;
        }

        private void butterworthToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            Image<Gray, float> im_pad = filter(img.Convert<Gray, float>());
            Image<Gray, float> retimage = new Image<Gray, float>(2 * img.Width, 2 * img.Height);
            h = retbutterworthmask(1, 1);
            retimage = convolve(im_pad, h);
            imageBox1.Image = retimage;
        }

        private void gaussianToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            Image<Gray, float> im_pad = filter(img.Convert<Gray, float>());
            Image<Gray, float> retimage = new Image<Gray, float>(2 * img.Width, 2 * img.Height);
            h = retgaussianmask(1);
            retimage = convolve(im_pad, h);
            imageBox1.Image = retimage;
        }

    }
}
