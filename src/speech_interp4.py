import sys, os
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(this_dir + '/../magphase/src'))
import numpy as np
from matplotlib import pyplot as plt
import libutils as lu
import libaudio as la
import magphase as mp
from scikits.talkbox import lpc
from scipy.signal import lfilter
from scipy import interpolate

def lpc_to_mag(v_lpc, fft_len=4096):
    '''
    Computed the magnitude spectrum from LPC coefficients using approximation by FFT method.
    '''
    v_imp = np.r_[1, np.zeros(fft_len-1)]
    v_imp_filt = lfilter(np.array([1.0]), v_lpc, v_imp)

    v_mag = np.absolute(np.fft.fft(v_imp_filt))

    v_mag = la.remove_hermitian_half(v_mag[None,:])[0]

    return v_mag


def get_formant_locations_from_spec_env(v_sp_env):
    '''
    v_sp_env could be in db, log, or absolute value.
    '''

    v_mag_diff = np.diff(v_sp_env)
    v_mag_diff[v_mag_diff>=0.0] = 1.0
    v_mag_diff[v_mag_diff<0.0] = -1.0
    v_mag_diff_diff = np.diff(v_mag_diff)

    v_frmnts_bins  = np.where(v_mag_diff_diff<0.0)[0] + 1
    v_frmnts_gains = v_sp_env[v_frmnts_bins]
    return v_frmnts_bins, v_frmnts_gains


def get_formant_locations_from_raw_long_frame(wavfile, nx, fft_len):
    '''
    nx: frame index
    '''

    v_sig, fs = la.read_audio_file(wavfile)

    # Epoch detection:
    v_pm_sec, v_voi = la.reaper_epoch_detection(wavfile)
    v_pm = lu.round_to_int(v_pm_sec * fs)

    # Raw-long Frame extraction:

    v_frm_long = v_sig[v_pm[nx-2]:v_pm[nx+2]+1]

    # Win:
    left_len  = v_pm[nx] - v_pm[nx-2]
    right_len = v_pm[nx+2] - v_pm[nx]
    v_win = la.gen_non_symmetric_win(left_len, right_len, np.hanning, b_norm=False)
    v_frm_long_win = v_frm_long * v_win


    # Spectrum:
    v_mag = np.absolute(np.fft.fft(v_frm_long_win, n=fft_len))
    v_mag_db = la.db(la.remove_hermitian_half(v_mag[None,:])[0])

    # Formant extraction -LPC method:--------------------------------------------------
    v_lpc, v_e, v_refl = lpc(v_frm_long_win, 120)

    b_use_lpc_roots = False
    if b_use_lpc_roots:
        v_lpc_roots = np.roots(v_lpc)
        v_lpc_angles = np.angle(v_lpc_roots)
        v_lpc_angles = v_lpc_angles[v_lpc_angles>=0]
        v_lpc_angles = np.sort(v_lpc_angles)
        fft_len_half = 1 + fft_len / 2
        v_lpc_roots_bins = v_lpc_angles * fft_len_half / np.pi

    v_lpc_mag = lpc_to_mag(v_lpc, fft_len=fft_len)
    v_lpc_mag_db = la.db(v_lpc_mag)
    v_lpc_mag_db = v_lpc_mag_db - np.mean(v_lpc_mag_db) + np.mean(v_mag_db)

    v_frmnts_bins, v_frmnts_gains_db = get_formant_locations_from_spec_env(v_lpc_mag_db)

    # Getting bandwidth:
    fft_len_half = 1 + fft_len / 2
    v_vall_bins = get_formant_locations_from_spec_env(-v_lpc_mag_db)[0]
    v_vall_bins = np.r_[0, v_vall_bins, fft_len_half-1]

    nfrmnts = v_frmnts_bins.size
    v_frmnts_bw = np.zeros(nfrmnts) - 1.0
    for nx_f in xrange(nfrmnts):
        #Left slope:
        curr_frmnt_bin  = v_frmnts_bins[nx_f]
        curr_vall_l_bin = v_vall_bins[nx_f]
        curr_vall_r_bin = v_vall_bins[nx_f+1]

        curr_midp_l = int((curr_frmnt_bin + curr_vall_l_bin) / 2.0)
        curr_midp_r = int((curr_frmnt_bin + curr_vall_r_bin) / 2.0)

        slope_l = (v_frmnts_gains_db[nx_f] - v_lpc_mag_db[curr_midp_l]) / (v_frmnts_bins[nx_f] - curr_midp_l).astype(float)
        slope_r = (v_frmnts_gains_db[nx_f] - v_lpc_mag_db[curr_midp_r]) / (v_frmnts_bins[nx_f] - curr_midp_r).astype(float)

        slope_ave = (slope_l - slope_r) / 2.0

        v_frmnts_bw[nx_f] = 1.0 / slope_ave

    # Filtering by bandwidth:
    bw_thress         = 7.0
    v_frmnts_bins     = v_frmnts_bins[v_frmnts_bw<bw_thress]
    v_frmnts_gains_db = v_frmnts_gains_db[v_frmnts_bw<bw_thress]
    v_frmnts_bw       = v_frmnts_bw[v_frmnts_bw<bw_thress]

    # Computing frame short:--------------------------------
    # Win:
    left_len_short  = v_pm[nx] - v_pm[nx-1]
    right_len_short = v_pm[nx+1] - v_pm[nx]
    v_win_short = la.gen_non_symmetric_win(left_len_short, right_len_short, np.hanning, b_norm=False)
    v_frm_short = v_sig[v_pm[nx-1]:v_pm[nx+1]+1]
    v_frm_short_win = v_frm_short * v_win_short
    shift = v_pm[nx] - v_pm[nx-1]

    # Formant extraction - True envelope method:----------------------------------------
    # Not finished.
    #v_true_env_db = la.true_envelope(v_mag_db[None,:], in_type='db', ncoeffs=400, thres_db=0.1)[0]

    if False:
        plt.figure(); plt.plot(v_mag_db); plt.plot(v_lpc_mag_db); plt.grid(); plt.show()



    return v_mag_db, v_lpc_mag_db, v_frmnts_bins, v_frmnts_gains_db, v_frmnts_bw, v_frm_short_win, shift



def formant_mapping(v_frmnts_bins_a, v_frmnts_gains_db_a, v_frmnts_bins_b, v_frmnts_gains_db_b, fft_len):

    # Order according to gain:
    v_order_gain_a = np.argsort(-v_frmnts_gains_db_a)
    v_order_gain_b = np.argsort(-v_frmnts_gains_db_b)

    v_frmnts_bins_ord_gain_a     = v_frmnts_bins_a[v_order_gain_a]
    v_frmnts_gains_db_ord_gain_a = v_frmnts_gains_db_a[v_order_gain_a]

    v_frmnts_bins_ord_gain_b     = v_frmnts_bins_b[v_order_gain_b]
    v_frmnts_gains_db_ord_gain_b = v_frmnts_gains_db_b[v_order_gain_b]

    nfrmnts = np.minimum(v_frmnts_bins_a.size, v_frmnts_bins_b.size)
    v_frmnts_bins_ord_gain_b_dyn = v_frmnts_bins_ord_gain_b.copy()
    v_map_a_to_b_ord_gain = np.zeros(nfrmnts, dtype='int') - 1
    for nx_f in xrange(nfrmnts):
        v_diffs = np.abs(v_frmnts_bins_ord_gain_a[nx_f] - v_frmnts_bins_ord_gain_b_dyn)
        nx_chosen_b_frmnt = np.argmin(v_diffs)
        v_map_a_to_b_ord_gain[nx_f] = nx_chosen_b_frmnt
        v_frmnts_bins_ord_gain_b_dyn[nx_chosen_b_frmnt] = -fft_len  # A really big number

    # Cut down unnecessary elemnts:
    v_frmnts_bins_ord_gain_a = v_frmnts_bins_ord_gain_a[:nfrmnts]
    v_frmnts_gains_db_ord_gain_a = v_frmnts_gains_db_ord_gain_a[:nfrmnts]

    # Reorder b vectors to match a:
    v_frmnts_bins_b_match_a_ord_gain     = v_frmnts_bins_ord_gain_b[v_map_a_to_b_ord_gain]
    v_frmnts_gains_db_b_match_a_ord_gain = v_frmnts_gains_db_ord_gain_b[v_map_a_to_b_ord_gain]

    # Order by frequency bins:
    v_order_bins_a = np.argsort(v_frmnts_bins_ord_gain_a)
    v_frmnts_bins_a = v_frmnts_bins_ord_gain_a[v_order_bins_a]
    v_frmnts_bins_b_match_a = v_frmnts_bins_b_match_a_ord_gain[v_order_bins_a]


    v_dists_bins = np.abs(v_frmnts_bins_a - v_frmnts_bins_b_match_a)
    # Order according to frequency bin:
    for nx_f in xrange(nfrmnts):
        curr_bin_a = v_frmnts_bins_a[nx_f]
        curr_bin_b = v_frmnts_bins_b_match_a[nx_f]

        if curr_bin_a==-1:
            continue

        # Iteration per next locations:
        for nx_f2 in xrange(nx_f+1, nfrmnts):
            curr_bin_b2 = v_frmnts_bins_b_match_a[nx_f2]
            # Si se cruzan:
            if curr_bin_b2 < curr_bin_b:
                # Si el 2 es mas largo (remove 2):
                if v_dists_bins[nx_f2] > v_dists_bins[nx_f]:
                    v_frmnts_bins_a[nx_f2] = -1

                # Si el 1 es mas largo:
                else:
                    v_frmnts_bins_a[nx_f] = -1
                    continue

    v_nx_frmnts_filt =  np.where(v_frmnts_bins_a >= 0)[0]
    v_frmnts_bins_a_filt = v_frmnts_bins_a[v_nx_frmnts_filt]
    v_frmnts_bins_b_match_a_filt = v_frmnts_bins_b_match_a[v_nx_frmnts_filt]
    # 7,9,10,19,20,27

    # Debug:
    if False:
        plt.figure()
        plt.plot(v_lpc_mag_db_a)
        plt.plot(v_lpc_mag_db_b)
        for nx_f in xrange(v_map_a_to_b_ord_gain.size):
            plt.plot(np.r_[v_frmnts_bins_ord_gain_a[nx_f], v_frmnts_bins_ord_gain_b[v_map_a_to_b_ord_gain[nx_f]]], np.r_[v_frmnts_gains_db_ord_gain_a[nx_f], v_frmnts_gains_db_ord_gain_b[v_map_a_to_b_ord_gain[nx_f]]], 'k')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(v_lpc_mag_db_a)
        plt.plot(v_lpc_mag_db_b)
        for nx_f in xrange(v_frmnts_bins_a_filt.size):
            v_curr_x = np.r_[v_frmnts_bins_a_filt[nx_f], v_frmnts_bins_b_match_a_filt[nx_f]]
            v_curr_y = np.r_[ v_lpc_mag_db_a[v_frmnts_bins_a_filt[nx_f]],  v_lpc_mag_db_b[v_frmnts_bins_b_match_a_filt[nx_f]]]
            plt.plot(v_curr_x, v_curr_y, 'k')
        plt.grid()
        plt.show()

    return v_frmnts_bins_a_filt, v_frmnts_bins_b_match_a_filt


def warp_mag_spec(v_lpc_mag_db_a, v_frmnts_bins_a_filt, v_frmnts_bins_b_filt, fft_len, sp_weight):

    # Weighting:
    v_targ_frmnts_bins = v_frmnts_bins_a_filt * (1.0 - sp_weight) + v_frmnts_bins_b_filt * sp_weight

    # Generate warping function:
    fft_len_half = 1 + fft_len / 2
    func_intrp = interpolate.interp1d(np.r_[0, v_frmnts_bins_a_filt, fft_len_half-1], np.r_[0, v_targ_frmnts_bins, fft_len_half-1], bounds_error=True, axis=0, kind='slinear')
    v_lin_bins = np.arange(fft_len_half)
    v_warp = func_intrp(v_lin_bins)
    # Protection:
    v_warp[-1] = v_lin_bins[-1]

    # Do the warping:
    func_intrp = interpolate.interp1d(v_warp, v_lpc_mag_db_a, bounds_error=True, axis=0, kind='slinear')
    v_lpc_mag_db_a_warp = func_intrp(v_lin_bins)

    return v_lpc_mag_db_a_warp


def fft_filter(v_frm_short_a, shift_a, v_spec_diff_db_a, fft_len):

    # dB to absolute:
    v_spec_diff_a = la.db(v_spec_diff_db_a, b_inv=True)

    right_a = v_frm_short_a.size - shift_a
    v_frm_short_a_ext = np.r_[np.zeros(fft_len/2 - shift_a) ,  v_frm_short_a , np.zeros(fft_len/2 - right_a)]
    v_fft_frm_short_a_ext = np.fft.fft(v_frm_short_a_ext) * la.add_hermitian_half(v_spec_diff_a[None,:], data_type='mag')[0]

    # To time domain:
    v_frm_short_a_ext_filt = np.fft.ifft(v_fft_frm_short_a_ext).real

    return v_frm_short_a_ext_filt


def speech_interp(wavfile_a, wavfile_b, nx_strt_a, nx_strt_b, nframes, fft_len):




    # Get formants:
    v_mag_db_a, v_lpc_mag_db_a, v_frmnts_bins_a, v_frmnts_gains_db_a, v_frmnts_bw_a, v_frm_short_a, shift_a = get_formant_locations_from_raw_long_frame(wavfile_a, nx_a, fft_len)
    v_mag_db_b, v_lpc_mag_db_b, v_frmnts_bins_b, v_frmnts_gains_db_b, v_frmnts_bw_b, v_frm_short_b, shift_b = get_formant_locations_from_raw_long_frame(wavfile_b, nx_b, fft_len)

    # Formant mapping:----------------------------------------------------------------
    v_frmnts_bins_a_filt, v_frmnts_bins_b_filt = formant_mapping(v_frmnts_bins_a, v_frmnts_gains_db_a, v_frmnts_bins_b, v_frmnts_gains_db_b, fft_len)

    # Warping:---------------------------------------------------------------------

    # True envelope:
    v_true_env_db_a = la.true_envelope(v_mag_db_a[None,:], in_type='db', ncoeffs=400, thres_db=0.1)[0]
    v_true_env_db_b = la.true_envelope(v_mag_db_b[None,:], in_type='db', ncoeffs=400, thres_db=0.1)[0]

    v_sp_env_db_a_warp = warp_mag_spec(v_true_env_db_a, v_frmnts_bins_a_filt, v_frmnts_bins_b_filt, fft_len, sp_weight)
    v_sp_env_db_b_warp = warp_mag_spec(v_true_env_db_b, v_frmnts_bins_b_filt, v_frmnts_bins_a_filt, fft_len, (1-sp_weight))

    #v_sp_env_db_a_warp = warp_mag_spec(v_lpc_mag_db_a, v_frmnts_bins_a_filt, v_frmnts_bins_b_filt, fft_len, sp_weight)
    #v_sp_env_db_b_warp = warp_mag_spec(v_lpc_mag_db_b, v_frmnts_bins_b_filt, v_frmnts_bins_a_filt, fft_len, (1-sp_weight))

    # Spectral envelope mix:-------------------------------------------------------
    v_sp_env_db_targ = v_sp_env_db_a_warp * (1.0-sp_weight) + v_sp_env_db_b_warp * sp_weight

    # Source mix:------------------------------------------------------------------
    v_spec_diff_db_a = v_sp_env_db_targ - v_true_env_db_a
    v_spec_diff_db_b = v_sp_env_db_targ - v_true_env_db_b

    # Filtering (FFT filter):
    v_frm_short_a_ext_filt = fft_filter(v_frm_short_a, shift_a, v_spec_diff_db_a, fft_len)
    v_frm_short_b_ext_filt = fft_filter(v_frm_short_b, shift_b, v_spec_diff_db_b, fft_len)

    # Mix:
    v_frm_short_ext_filt = v_frm_short_a_ext_filt * (1.0-sp_weight) + v_frm_short_b_ext_filt * sp_weight


    return



if __name__ == '__main__':

    # INPUT:=====================================================================================================
    fft_len = 4096
    wavfile_a = '/home/felipe/Cloud/Education/UoE/Projects/speech_interp/data/wav/sim_48k/a_p0_1_shorter_48k.wav'
    wavfile_b = '/home/felipe/Cloud/Education/UoE/Projects/speech_interp/data/wav/sim_48k/o_p0_1_shorter_48k.wav'

    nx_a = 30
    nx_b = 30

    sp_weight = 0.5

    # PROCESS:====================================================================================================

    v_sig_interp = speech_interp(wavfile_a, wavfile_b, nx_strt_a, nx_strt_b, nframes, fft_len)





    # Get formants:
    v_mag_db_a, v_lpc_mag_db_a, v_frmnts_bins_a, v_frmnts_gains_db_a, v_frmnts_bw_a, v_frm_short_a, shift_a = get_formant_locations_from_raw_long_frame(wavfile_a, nx_a, fft_len)
    v_mag_db_b, v_lpc_mag_db_b, v_frmnts_bins_b, v_frmnts_gains_db_b, v_frmnts_bw_b, v_frm_short_b, shift_b = get_formant_locations_from_raw_long_frame(wavfile_b, nx_b, fft_len)

    # Formant mapping:----------------------------------------------------------------
    v_frmnts_bins_a_filt, v_frmnts_bins_b_filt = formant_mapping(v_frmnts_bins_a, v_frmnts_gains_db_a, v_frmnts_bins_b, v_frmnts_gains_db_b, fft_len)

    # Warping:---------------------------------------------------------------------

    # True envelope:
    v_true_env_db_a = la.true_envelope(v_mag_db_a[None,:], in_type='db', ncoeffs=400, thres_db=0.1)[0]
    v_true_env_db_b = la.true_envelope(v_mag_db_b[None,:], in_type='db', ncoeffs=400, thres_db=0.1)[0]

    v_sp_env_db_a_warp = warp_mag_spec(v_true_env_db_a, v_frmnts_bins_a_filt, v_frmnts_bins_b_filt, fft_len, sp_weight)
    v_sp_env_db_b_warp = warp_mag_spec(v_true_env_db_b, v_frmnts_bins_b_filt, v_frmnts_bins_a_filt, fft_len, (1-sp_weight))

    #v_sp_env_db_a_warp = warp_mag_spec(v_lpc_mag_db_a, v_frmnts_bins_a_filt, v_frmnts_bins_b_filt, fft_len, sp_weight)
    #v_sp_env_db_b_warp = warp_mag_spec(v_lpc_mag_db_b, v_frmnts_bins_b_filt, v_frmnts_bins_a_filt, fft_len, (1-sp_weight))

    # Spectral envelope mix:-------------------------------------------------------
    v_sp_env_db_targ = v_sp_env_db_a_warp * (1.0-sp_weight) + v_sp_env_db_b_warp * sp_weight

    # Source mix:------------------------------------------------------------------
    v_spec_diff_db_a = v_sp_env_db_targ - v_true_env_db_a
    v_spec_diff_db_b = v_sp_env_db_targ - v_true_env_db_b

    # Filtering (FFT filter):
    v_frm_short_a_ext_filt = fft_filter(v_frm_short_a, shift_a, v_spec_diff_db_a, fft_len)
    v_frm_short_b_ext_filt = fft_filter(v_frm_short_b, shift_b, v_spec_diff_db_b, fft_len)

    # Mix:
    v_frm_short_ext_filt = v_frm_short_a_ext_filt * (1.0-sp_weight) + v_frm_short_b_ext_filt * sp_weight





    if True:
        plt.figure(); plt.plot(v_frm_short_a_ext_filt); plt.plot(v_frm_short_b_ext_filt); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_frm_short_a_ext_filt); plt.plot(v_frm_short_b_ext_filt); plt.plot(v_frm_short_ext_filt); plt.grid(); plt.show()



    if False:
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(5 + v_lpc_mag_db_a); plt.plot(v_sp_env_db_targ); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(6.0 + v_lpc_mag_db_a); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(v_mag_db_a - v_lpc_mag_db_a); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(v_true_env_db_a); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(v_mag_db_a - v_true_env_db_a); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(v_mag_db_a - v_true_env_db_a); plt.plot(v_mag_db_a - v_lpc_mag_db_a); plt.grid(); plt.show()

        plt.figure(); plt.plot(v_frm_short_a_ext); plt.plot(v_frm_short_a_ext_filt); plt.grid(); plt.show()












    if False:
        plt.figure()
        plt.plot(v_lpc_mag_db_a)
        plt.plot(v_lpc_mag_db_b)
        #plt.plot(v_warp, v_lpc_mag_db_a)
        plt.plot(v_sp_env_db_a_warp)
        plt.plot(v_sp_env_db_b_warp)
        plt.grid()
        plt.show()



        plt.figure(); plt.plot(v_sp_env_db_a_warp); plt.plot(v_sp_env_db_b_warp); plt.plot(v_sp_env_db_targ); plt.grid(); plt.show()
        plt.figure(); plt.plot(v_lpc_mag_db_a); plt.plot(v_sp_env_db_targ); plt.plot(v_spec_diff_a); plt.grid(); plt.show()