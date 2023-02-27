import itertools

import numpy as np


__all__ = ['ComplexModulation', 'QAModulation']

def _int2binlist(int_, width=None):
    if width is None:
        width = max(int_.bit_length(), 1)
    return [(int_ >> i) & 1 for i in range(width)] 


def int2binlist(int_, width=None):
    """
    Converts an integer to its bit array representation.
    """
    return np.array(_int2binlist(int_, width)) 

class _Modulation:
    def __init__(self, constellation, labeling):
        self._constellation = np.array(constellation)
        self._order = self._constellation.size
        if self._order & (self._order - 1):
            raise ValueError("The length of constellation must be a power of two")
        self._bits_per_symbol = (self._order - 1).bit_length()

        self._labeling = np.array(labeling, dtype=np.int)
        if not np.array_equal(np.sort(self._labeling), np.arange(self._order)):
            raise ValueError("The labeling must be a permutation of [0 : order)")
        self._mapping = {symbol: tuple(int2binlist(label, width=self._bits_per_symbol)) \
                         for (symbol, label) in enumerate(self._labeling)}
        self._inverse_mapping = dict((value, key) for key, value in self._mapping.items())

        self._channel_snr = 1.0
        self._channel_N0 = self.energy_per_symbol / self._channel_snr

    def __repr__(self):
        args = 'constellation={}, labeling={}'.format(self._constellation.tolist(), self._labeling.tolist())
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def constellation(self):
        """
        The constellation :math:`\\mathcal{S}` of the modulation. This property is read-only.
        """
        return self._constellation

    @property
    def labeling(self):
        """
        The binary labeling :math:`\\mathcal{Q}` of the modulation. This property is read-only.
        """
        return self._labeling

    @property
    def order(self):
        """
        The order :math:`M` of the modulation. This property is read-only.
        """
        return self._order

    @property
    def bits_per_symbol(self):
        """
        The number :math:`m` of bits per symbol of the modulation. It is given by :math:`m = \\log_2 M`, where :math:`M` is the order of the modulation. This property is read-only.
        """
        return self._bits_per_symbol

    @property
    def energy_per_symbol(self):
        """
        The average symbol energy :math:`E_\\mathrm{s}` of the constellation. It assumes equiprobable symbols. It is given by

        .. math::

            E_\\mathrm{s} = \\frac{1}{M} \\sum_{s_i \\in \\mathcal{S}} |s_i|^2,

        where :math:`|s_i|^2` is the energy of symbol :math:`s_i \\in \\mathcal{S}` and :math:`M` is the order of the modulation. This property is read-only.
        """
        return np.real(np.dot(self._constellation, self._constellation.conj())) / self._order

    @property
    def energy_per_bit(self):
        """
        The average bit energy :math:`E_\\mathrm{b}` of the constellation. It assumes equiprobable symbols. It is given by :math:`E_\\mathrm{b} = E_\\mathrm{s} / m`, where :math:`E_\\mathrm{s}` is the average symbol energy, and :math:`m` is the number of bits per symbol of the modulation.
        """
        return self.energy_per_symbol / np.log2(self._order)

    @property
    def minimum_distance(self):
        """
        The minimum euclidean distance of the constellation.
        """
        pass

    @property
    def channel_snr(self):
        """
        The signal-to-noise ratio :math:`\\mathrm{SNR}` of the channel. This is used in soft-decision methods. This is a read-and-write property.
        """
        return self._channel_snr

    @channel_snr.setter
    def channel_snr(self, value):
        self._channel_snr = value
        self._channel_N0 = self.energy_per_symbol / self._channel_snr

    def bits_to_symbols(self, bits):
        """
        Converts bits to symbols using the modulation binary labeling.

        .. rubric:: Input

        :code:`bits` : 1D-array of :obj:`int`
            The bits to be converted. It should be a 1D-array of integers in the set :math:`\\{ 0, 1 \\}`. Its length must be a multiple of :math:`m`.

        .. rubric:: Output

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols corresponding to :code:`bits`. It is a 1D-array of integers in the set :math:`[0 : M)`. Its length is equal to the length of :code:`bits` divided by :math:`m`.
        """
        m = self._bits_per_symbol
        n_symbols = len(bits) // m
        assert len(bits) == n_symbols * m
        symbols = np.empty(n_symbols, dtype=np.int)
        for i, bit_sequence in enumerate(np.reshape(bits, newshape=(n_symbols, m))):
            symbols[i] = self._inverse_mapping[tuple(bit_sequence)]
        return symbols

    def symbols_to_bits(self, symbols):
        """
        Converts symbols to bits using the modulation binary labeling.

        .. rubric:: Input

        :code:`symbols` : 1D-array of :obj:`int`
            The symbols to be converted. It should be a 1D-array of integers in the set :math:`[0 : M)`. It may be of any length.

        .. rubric:: Output

        :code:`bits` : 1D-array of :obj:`int`
            The bits corresponding to :code:`symbols`. It is a 1D-array of integers in the set :math:`\\{ 0, 1 \\}`. Its length is equal to the length of :code:`symbols` multiplied by :math:`m`.
        """
        m = self._bits_per_symbol
        n_bits = len(symbols) * m
        bits = np.empty(n_bits, dtype=np.int)
        for i, symbol in enumerate(symbols):
            bits[i*m : (i + 1)*m] = self._mapping[symbol]
        return bits

    def modulate(self, bits):
        """
        Modulates a sequence of bits to its corresponding constellation points.
        """
        symbols = self.bits_to_symbols(bits)
        return self._constellation[symbols]

    def _hard_symbol_demodulator(self, received):
        """General minimum distance hard demodulator."""
        mpoints, mconst = np.meshgrid(received, self._constellation)
        return np.argmin(np.absolute(mpoints - mconst), axis=0)

    def _soft_bit_demodulator(self, received):
        """Computes L-values of received points"""
        m = self._bits_per_symbol

        def pdf_received_given_bit(bit_index, bit_value):
            bits = np.empty(m, dtype=np.int)
            bits[bit_index] = bit_value
            rest_index = np.setdiff1d(np.arange(m), [bit_index])
            f = 0.0
            for b_rest in itertools.product([0, 1], repeat=m-1):
                bits[rest_index] = b_rest
                point = self._constellation[self._inverse_mapping[tuple(bits)]]
                f += np.exp(-np.abs(received - point)**2 / self._channel_N0)
            return f

        soft_bits = np.empty(len(received)*m, dtype=np.float)
        for bit_index in range(m):
            p0 = pdf_received_given_bit(bit_index, 0)
            p1 = pdf_received_given_bit(bit_index, 1)
            soft_bits[bit_index::m] = np.log(p0 / p1)

        return soft_bits

    def demodulate(self, received, decision_method='hard'):
        """
        Demodulates a sequence of received points to a sequence of bits.
        """
        if decision_method == 'hard':
            symbols_hat = self._hard_symbol_demodulator(received)
            return self.symbols_to_bits(symbols_hat)
        elif decision_method == 'soft':
            return self._soft_bit_demodulator(received)
        else:
            raise ValueError("Parameter 'decision_method' should be either 'hard' or 'soft'")

    @staticmethod
    def _labeling_natural(order):
        labeling = np.arange(order)
        return labeling

    @staticmethod
    def _labeling_reflected(order):
        labeling = np.arange(order)
        labeling ^= (labeling >> 1)
        return labeling

    @staticmethod
    def _labeling_reflected_2d(order_I, order_Q):
        labeling_I = _Modulation._labeling_reflected(order_I)
        labeling_Q = _Modulation._labeling_reflected(order_Q)
        labeling = np.empty(order_I * order_Q, dtype=np.int)
        for i, (i_Q, i_I) in enumerate(itertools.product(labeling_Q, labeling_I)):
            labeling[i] = i_I + order_I * i_Q
        return labeling


class ComplexModulation(_Modulation):
    """
    General complex modulation scheme. A *complex modulation scheme* of order :math:`M` is defined by a *constellation* :math:`\\mathcal{S}`, which is an ordered subset (a list) of complex numbers, with :math:`|\\mathcal{S}| = M`, and a *binary labeling* :math:`\\mathcal{Q}`, which is a permutation of :math:`[0: M)`. The order :math:`M` of the modulation must be a power of :math:`2`.
    """
    def __init__(self, constellation, labeling):
        """
        Constructor for the class. It expects the following parameters:

        :code:`constellation` : 1D-array of :obj:`complex`
            The constellation :math:`\\mathcal{S}` of the modulation. Must be a 1D-array containing :math:`M` complex numbers.

        :code:`labeling` : 1D-array of :obj:`int`
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Must be a 1D-array of integers corresponding to a permutation of :math:`[0 : M)`.

        .. rubric:: Examples

        >>> mod = komm.ComplexModulation(constellation=[0.0, -1, 1, 1j], labeling=[0, 1, 2, 3])
        >>> mod.constellation
        array([ 0.+0.j, -1.+0.j,  1.+0.j,  0.+1.j])
        >>> mod.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        array([ 0.+0.j,  0.+1.j,  0.+0.j, -1.+0.j, -1.+0.j])
        """
        super().__init__(np.array(constellation, dtype=np.complex), labeling)



class QAModulation(ComplexModulation):
    """
    Quadratude-amplitude modulation (QAM). It is a complex modulation scheme (:class:`ComplexModulation`) in which the constellation :math:`\\mathcal{S}` is given as a cartesian product of two PAM (:class:`PAModulation`) constellations, namely, the *in-phase constellation*, and the *quadrature constellation*. More precisely,

    .. math::
        \\mathcal{S} = \\{ [\\pm(2i_\\mathrm{I} + 1)A_\\mathrm{I} \\pm \\mathrm{j}(2i_\\mathrm{Q} + 1)A_\\mathrm{Q}] \\exp(\\mathrm{j}\\phi) : i_\\mathrm{I} \\in [0 : M_\\mathrm{I}), i_\\mathrm{Q} \\in [0 : M_\\mathrm{Q}) \\},

    where :math:`M_\\mathrm{I}` and :math:`M_\\mathrm{Q}` are the *orders* (powers of :math:`2`), and :math:`A_\\mathrm{I}` and :math:`A_\\mathrm{Q}` are the *base amplitudes* of the in-phase and quadratude constellations, respectively. Also, :math:`\\phi` is the *phase offset*. The size of the resulting complex-valued constellation is :math:`M = M_\\mathrm{I} M_\\mathrm{Q}`, a power of :math:`2`. The QAM constellation is depicted below for :math:`(M_\\mathrm{I}, M_\\mathrm{Q}) = (4, 4)` with :math:`A_\\mathrm{I} = A_\\mathrm{Q} = A`, and for :math:`(M_\\mathrm{I}, M_\\mathrm{Q}) = (4, 2)` with :math:`A_\\mathrm{I} = A` and :math:`A_\\mathrm{Q} = 2A`.

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/qam_16.png
       :alt: 16-QAM constellation

    .. |fig2| image:: figures/qam_8.png
       :alt: 8-QAM constellation

    .. |quad| unicode:: 0x2001
       :trim:
    """
    def __init__(self, orders, base_amplitudes=1.0, phase_offset=0.0, labeling='reflected_2d'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`orders` : :obj:`(int, int)` or :obj:`int`
            A tuple :math:`(M_\\mathrm{I}, M_\\mathrm{Q})` with the orders of the in-phase and quadrature constellations, respectively; both :math:`M_\\mathrm{I}` and :math:`M_\\mathrm{Q}` must be powers of :math:`2`. If specified as a single integer :math:`M`, then it is assumed that :math:`M_\\mathrm{I} = M_\\mathrm{Q} = \\sqrt{M}`; in this case, :math:`M` must be an square power of :math:`2`.

        :code:`base_amplitudes` : :obj:`(float, float)` or :obj:`float`, optional
            A tuple :math:`(A_\\mathrm{I}, A_\\mathrm{Q})` with the base amplitudes of the in-phase and quadrature constellations, respectively.  If specified as a single float :math:`A`, then it is assumed that :math:`A_\\mathrm{I} = A_\\mathrm{Q} = A`. The default value is :math:`1.0`.

        :code:`phase_offset` : :obj:`float`, optional
            The phase offset :math:`\\phi` of the constellation. The default value is :code:`0.0`.

        :code:`labeling` : (1D-array of :obj:`int`) or :obj:`str`, optional
            The binary labeling :math:`\\mathcal{Q}` of the modulation. Can be specified either as a 1D-array of integers, in which case must be permutation of :math:`[0 : M)`, or as a string, in which case must be one of :code:`'natural'` or :code:`'reflected_2d'`. The default value is :code:`'reflected_2d'` (Gray code).

        .. rubric:: Examples

        >>> qam = komm.QAModulation(16)
        >>> qam.constellation  #doctest: +NORMALIZE_WHITESPACE
        array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
               -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
               -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
               -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
        >>> qam.labeling
        array([ 0,  1,  3,  2,  4,  5,  7,  6, 12, 13, 15, 14,  8,  9, 11, 10])
        >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0])
        array([-3.+1.j, -3.-1.j])

        >>> qam = komm.QAModulation(orders=(4, 2), base_amplitudes=(1.0, 2.0))
        >>> qam.constellation
        array([-3.-2.j, -1.-2.j,  1.-2.j,  3.-2.j, -3.+2.j, -1.+2.j,  1.+2.j,
                3.+2.j])
        >>> qam.labeling
        array([0, 1, 3, 2, 4, 5, 7, 6])
        >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0, 1])
        array([-3.+2.j, -1.-2.j, -1.+2.j])
        """
        if isinstance(orders, (tuple, list)):
            order_I, order_Q = int(orders[0]), int(orders[1])
            self._orders = (order_I, order_Q)
        else:
            order_I = order_Q = int(np.sqrt(orders))
            self._orders = int(orders)

        if isinstance(base_amplitudes, (tuple, list)):
            base_amplitude_I, base_amplitude_Q = float(base_amplitudes[0]), float(base_amplitudes[1])
            self._base_amplitudes = (base_amplitude_I, base_amplitude_Q)
        else:
            base_amplitude_I = base_amplitude_Q = float(base_amplitudes)
            self._base_amplitudes = base_amplitude_I

        constellation_I = base_amplitude_I * np.arange(-order_I + 1, order_I, step=2, dtype=np.int)
        constellation_Q = base_amplitude_Q * np.arange(-order_Q + 1, order_Q, step=2, dtype=np.int)
        constellation = (constellation_I + 1j*constellation_Q[np.newaxis].T).flatten() * np.exp(1j * phase_offset)

        if isinstance(labeling, str):
            if labeling == 'natural':
                labeling = _Modulation._labeling_natural(order_I * order_Q)
            elif labeling == 'reflected_2d':
                labeling = _Modulation._labeling_reflected_2d(order_I, order_Q)
            else:
                raise ValueError("Only 'natural' or 'reflected_2d' are supported for {}".format(self.__class__.__name__))

        super().__init__(constellation, labeling)

        self._orders = orders
        self._base_amplitudes = base_amplitudes
        self._phase_offset = float(phase_offset)


    def __repr__(self):
        args = '{}, base_amplitudes={}, phase_offset={}'.format(self._orders, self._base_amplitudes, self._phase_offset)
        return '{}({})'.format(self.__class__.__name__, args)




def uniform_real_hard_demodulator(received, order):
    return np.clip(np.around((received + order - 1) / 2), 0, order - 1).astype(np.int)


def uniform_real_soft_bit_demodulator(received, snr):
    return -4 * snr * received


def ask_hard_demodulator(received, order):
    return np.clip(np.around((received.real + order - 1) / 2), 0, order - 1).astype(np.int)


def psk_hard_demodulator(received, order):
    phase_in_turns = np.angle(received) / (2 * np.pi)
    return np.mod(np.around(phase_in_turns * order).astype(np.int), order)


def bpsk_soft_bit_demodulator(received, snr):
    return 4 * snr * received.real


def qpsk_soft_bit_demodulator_reflected(received, snr):
    received_rotated = received * np.exp(2j * np.pi / 8)
    soft_bits = np.empty(2*received.size, dtype=np.float)
    soft_bits[0::2] = np.sqrt(8) * snr * received_rotated.real
    soft_bits[1::2] = np.sqrt(8) * snr * received_rotated.imag
    return soft_bits


def rectangular_hard_demodulator(received, order):
    L = int(np.sqrt(order))
    s_real = np.clip(np.around((received.real + L - 1) / 2), 0, L - 1).astype(np.int)
    s_imag = np.clip(np.around((received.imag + L - 1) / 2), 0, L - 1).astype(np.int)
    return s_real + L * s_imag