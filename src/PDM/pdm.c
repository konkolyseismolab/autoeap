#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RESTRICT restrict //for pointers
#define CONSTANT const    //for variables

double WEIGHT(double *RESTRICT w, int k){
  return w==NULL ? 1.0f : w[k];
}

double GAUSSIAN(double x){
  return expf(-0.5f *x*x);
}

double PHASE(double x, double f){
  return x * f - floor(x * f);
}

double phase_diff(
        CONSTANT double dt,
        CONSTANT double freq){
	double dphi = dt * freq - floorf(dt * freq);
	return ((dphi > 0.5f) ? 1.0f - dphi : dphi);
}

double var_step_function(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        CONSTANT double freq,
        CONSTANT int ndata,
        CONSTANT int nbins){
    double bin_means[nbins];
    double bin_wtots[nbins];
    int bin;
    double var_tot = 0.f;
    for (int i = 0; i < nbins; i++){
        bin_wtots[i] = 0.f;
        bin_means[i] = 0.f;
    }
    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * nbins);
        bin = bin % nbins;
        bin_wtots[bin] += w[i];
        bin_means[bin] += y[i] * w[i];
    }

    for(int i = 0; i < nbins; i++){
        if (bin_wtots[i] == 0.f)
            continue;
        bin_means[i] /= bin_wtots[i];
    }

    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * nbins);
        var_tot += w[i] * (y[i] - bin_means[bin]) * (y[i] - bin_means[bin]);
    }

    return var_tot;
}

double var_linear_interp(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        CONSTANT double freq,
        CONSTANT int ndata,
        CONSTANT int nbins){

    double bin_means[nbins];
    double bin_wtots[nbins];
    int bin, bin0, bin1;
    double var_tot = 0.f;
    double phase, y0, alpha;
    for(int i = 0; i < nbins; i++){
        bin_wtots[i] = 0.f;
        bin_means[i] = 0.f;
    }

    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * nbins);
        bin = bin % nbins;
        bin_wtots[bin] += w[i];
        bin_means[bin] += w[i] * y[i];
    }

    for (int i = 0; i < nbins; i++){
        if (bin_wtots[i] == 0.f)
            continue;
        bin_means[i] /= bin_wtots[i];
    }


    for (int i = 0; i < ndata; i++){
        phase = PHASE(t[i], freq);
        bin = (int) (phase * nbins);
        bin = bin % nbins;

        alpha = phase * nbins - floorf(phase * nbins) - 0.5f;
        bin0 = (alpha < 0) ? bin - 1 : bin;
        bin1 = (alpha < 0) ? bin : bin + 1;

        if (bin0 < 0)
            bin0 += nbins;
        if (bin1 >= nbins)
            bin1 -= nbins;

        alpha += (alpha < 0) ? 1.f : 0.f;
        y0 = (1.f - alpha) * bin_means[bin0] + alpha * bin_means[bin1];
        var_tot += w[i] * (y[i] - y0) * (y[i] - y0);
    }

    return var_tot;
}


double var_binless_tophat(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        CONSTANT double freq,
        CONSTANT int ndata,
        CONSTANT double dphi){
	double mbar, tj, wtot, var;
	double dph;
	var = 0.f;
	for(int j = 0; j < ndata; j++){
		mbar = 0.f;
		wtot = 0.f;
		tj = t[j];
		for(int k = 0; k < ndata; k++){
			dph = phase_diff(fabsf(t[k] - tj), freq);
      if(dph < dphi){
        wtot += w[k];
  			mbar += w[k] * y[k];
      }
		}
		mbar /= wtot;
		var += w[j] * (y[j] - mbar) * (y[j] - mbar);
	}
	return var;
}
double var_binless_gauss(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        CONSTANT double freq,
        CONSTANT int ndata,
        CONSTANT double dphi){
    double mbar, tj, wtot, var, wgt;
    var = 0.f;
    for(int j = 0; j < ndata; j++){
        mbar = 0.f;
        wtot = 0.f;
        tj = t[j];
        for(int k = 0; k < ndata; k++){
            double dphase = phase_diff(fabsf(t[k] - tj), freq);
            wgt   = w[k] * GAUSSIAN(dphase / dphi);
            mbar += wgt * y[k];
            wtot += wgt;
        }
        mbar /= wtot;
        var  += w[j] * (y[j] - mbar) * (y[j] - mbar);
    }
    return var;
}

double calc_var(
        double *RESTRICT y,
        double *RESTRICT w,
        CONSTANT int ndata){
    double ybar, var;
    ybar = 0.f;
    var = 0.f;
    // weights + weighted variance
    for(int i = 0; i < ndata; i++){
        ybar += w[i] * y[i];
    }
    for(int j = 0; j < ndata; j++){
      var += w[j] * (y[j] - ybar) * (y[j] - ybar);
    }
    return var;
}

void pdm_binless_tophat(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        double *RESTRICT freqs,
        double *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT double dphi,
        CONSTANT int nbins){
  double var;
  var = calc_var(y,w,ndata);
  for (int i=0; i<nfreqs; i++){
    power[i] = 1.f - var_binless_tophat(t, y, w, freqs[i], ndata, dphi) / var;
  }
}

void pdm_binless_gauss(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        double *RESTRICT freqs,
        double *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT double dphi,
        CONSTANT int nbins){
  double var;
  var = calc_var(y,w,ndata);
	for (int i=0; i<nfreqs; i++){
    power[i] = 1.f - var_binless_gauss(t, y, w, freqs[i], ndata, dphi) / var;
	}
}

void pdm_binned_linterp(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        double *RESTRICT freqs,
        double *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT double dphi,
        CONSTANT int nbins){
  double var;
  var = calc_var(y,w,ndata);
	for (int i=0; i<nfreqs; i++){
    power[i] = 1.f - var_linear_interp(t, y, w, freqs[i], ndata, nbins) / var;
	}
}
void pdm_binned_step(
        double *RESTRICT t,
        double *RESTRICT y,
        double *RESTRICT w,
        double *RESTRICT freqs,
        double *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT double dphi,
        CONSTANT int nbins){
  double var;
  var = calc_var(y,w,ndata);
	for (int i=0; i<nfreqs; i++){
    power[i] = 1.f - var_step_function(t, y, w, freqs[i], ndata, nbins) / var;
	}
}
