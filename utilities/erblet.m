function [c, fc] = erblet(path, verify_reconstruction, visualize)
    [f, fs] = audioread(path);
    [g, a, fc] = audfilters(fs, length(f), 'fractional');
    c = filterbank(f, {'realdual', g}, a);

    if verify_reconstruction
        r = 2*real(ifilterbank(c, g, a));
        err = norm(f-r);
        if err > 1e-10
            warning('Reconstruction error greater than 1e-10.\nError: %f', err)
        end
    end

    if visualize
        figure;
        plotfilterbank(c, a, fc, fs, 60, 'audtick');
        colorbar;
        title('ERBlet Filterbank Spectrogram');
        xlabel('Time (Sec)');
        ylabel('Center Frequency (Hz)'); 
    end
