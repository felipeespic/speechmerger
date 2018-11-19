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


    # Formant extraction - True envelope method:----------------------------------------
    # Not finished.
    #v_true_env_db = la.true_envelope(v_mag_db[None,:], in_type='db', ncoeffs=400, thres_db=0.1)[0]

    if False:
        plt.figure(); plt.plot(v_mag_db); plt.plot(v_lpc_mag_db); plt.grid(); plt.show()

    return v_frmnts_bins, v_frmnts_gains_db, v_mag_db,v_lpc_mag_db, v_frmnts_bw



if __name__ == '__main__':

    # INPUT:=====================================================================================================
    fft_len = 4096
    wavfile_a = '/home/felipe/Cloud/Education/UoE/Projects/speech_interp/data/wav/sim_48k/a_p0_1_shorter_48k.wav'
    wavfile_b = '/home/felipe/Cloud/Education/UoE/Projects/speech_interp/data/wav/sim_48k/o_p0_1_shorter_48k.wav'

    nx_a = 30
    nx_b = 30

    # PROCESS:====================================================================================================

    # Get formants:
    v_frmnts_bins_a, v_frmnts_gains_db_a, v_mag_db_a, v_lpc_mag_db_a, v_frmnts_bw_a = get_formant_locations_from_raw_long_frame(wavfile_a, nx_a, fft_len)
    v_frmnts_bins_b, v_frmnts_gains_db_b, v_mag_db_b, v_lpc_mag_db_b, v_frmnts_bw_b = get_formant_locations_from_raw_long_frame(wavfile_b, nx_b, fft_len)


    if False:
        plt.figure(); plt.plot(v_mag_db_a); plt.plot(v_lpc_mag_db_a); plt.plot(v_mag_db_b); plt.plot(v_lpc_mag_db_b); plt.grid(); plt.show()

    # Formant mapping:----------------------------------------------------------------

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

    # Reorder b vectors:
    v_frmnts_bins_b_match_a_ord_gain     = v_frmnts_bins_ord_gain_b[v_map_a_to_b_ord_gain]
    v_frmnts_gains_db_b_match_a_ord_gain = v_frmnts_gains_db_ord_gain_b[v_map_a_to_b_ord_gain]


    # Order according to frequency bin:
    v_order_bins_a = np.argsort(v_frmnts_bins_ord_gain_a)
    v_frmnts_bins_ord_bins_a = v_frmnts_bins_ord_gain_a[v_order_bins_a]
    v_frmnts_gains_db_ord_bins_a = v_frmnts_gains_db_ord_gain_a[v_order_bins_a]

    v_frmnts_bins_b_match_a_ord_bins = v_frmnts_bins_b_match_a_ord_gain[v_order_bins_a]
    v_frmnts_gains_db_b_match_a_ord_bins = v_frmnts_gains_db_b_match_a_ord_gain[v_order_bins_a]

    # Filter by crossing frequency:
    v_frmnts_bins_b_match_a_ord_bins_order = np.argsort(v_frmnts_bins_b_match_a_ord_bins)



    # Plot Mapping:
    if True:
        plt.figure()
        plt.plot(v_lpc_mag_db_a)
        plt.plot(v_lpc_mag_db_b)
        for nx_f in xrange(v_map_a_to_b_ord_gain.size):
            plt.plot(np.r_[v_frmnts_bins_ord_gain_a[nx_f], v_frmnts_bins_ord_gain_b[v_map_a_to_b_ord_gain[nx_f]]], np.r_[v_frmnts_gains_db_ord_gain_a[nx_f], v_frmnts_gains_db_ord_gain_b[v_map_a_to_b_ord_gain[nx_f]]], 'k')
        plt.grid()
        plt.show()







