# This code is from https://github.com/astropy/photutils
# Due to deprecation errors, I decided to import the parts relevant for autoeap here, to avoid unnecessarily high dependence on the package.

import numpy as np

from astropy.utils.exceptions import AstropyWarning

__all__ = ['NoDetectionsWarning']


class NoDetectionsWarning(AstropyWarning):
    """
    A warning class to indicate no sources were detected.
    """

class SegmentationImage:
    """
    Class for a segmentation image.
    Parameters
    ----------
    data : array_like (int)
        A segmentation array where source regions are labeled by
        different positive integer values.  A value of zero is reserved
        for the background.  The segmentation image must contain at
        least one non-zero pixel and must not contain any non-finite
        values (e.g. NaN, inf).
    """

    def __init__(self, data):
        self.data = data
        import numpy as np

    def __getitem__(self, index):
        return self.segments[index]

    def __iter__(self):
        for i in self.segments:
            yield i

    def __str__(self):
        cls_name = '<{0}.{1}>'.format(self.__class__.__module__,
                                      self.__class__.__name__)

        cls_info = []
        params = ['shape', 'nlabels', 'max_label']
        for param in params:
            cls_info.append((param, getattr(self, param)))
        fmt = ['{0}: {1}'.format(key, val) for key, val in cls_info]

        return '{}\n'.format(cls_name) + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __array__(self):
        """
        Array representation of the segmentation array (e.g., for
        matplotlib).
        """

        return self._data

    from astropy.utils import lazyproperty, deprecated
    @lazyproperty
    def _cmap(self):
        """
        A matplotlib colormap consisting of (random) muted colors.
        This is very useful for plotting the segmentation array.
        """

        return self.make_cmap(background_color='#000000', random_state=1234)

    @staticmethod
    def _get_labels(data):
        import numpy as np
        """
        Return a sorted array of the non-zero labels in the segmentation
        image.
        Parameters
        ----------
        data : array_like (int)
            A segmentation array where source regions are labeled by
            different positive integer values.  A value of zero is
            reserved for the background.
        Returns
        -------
        result : `~numpy.ndarray`
            An array of non-zero label numbers.
        Notes
        -----
        This is a static method so it can be used in
        :meth:`remove_masked_labels` on a masked version of the
        segmentation array.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm._get_labels(segm.data)
        array([1, 3, 4, 5, 7])
        """

        # np.unique also sorts elements
        return np.unique(data[data != 0])

    @lazyproperty
    def segments(self):
        """
        A list of `Segment` objects.
        The list starts with the *non-zero* label.  The returned list
        has a length equal to the number of labels and matches the order
        of the ``labels`` attribute.
        """

        segments = []
        for label, slc in zip(self.labels, self.slices):
            segments.append(Segment(self.data, label, slc,
                                    self.get_area(label)))
        return segments

    @property
    def data(self):
        """The segmentation array."""

        return self._data

    @data.setter
    def data(self, value):
        import numpy as np
        if np.any(~np.isfinite(value)):
            raise ValueError('data must not contain any non-finite values '
                             '(e.g. NaN, inf)')

        value = np.asarray(value, dtype=int)
        if not np.any(value):
            raise ValueError('The segmentation image must contain at least '
                             'one non-zero pixel.')

        if np.min(value) < 0:
            raise ValueError('The segmentation image cannot contain '
                             'negative integers.')

        if '_data' in self.__dict__:
            # needed only when data is reassigned, not on init
            self.__dict__ = {}

        self._data = value  # pylint: disable=attribute-defined-outside-init

    @lazyproperty
    def data_ma(self):
        import numpy as np
        """
        A `~numpy.ma.MaskedArray` version of the segmentation array
        where the background (label = 0) has been masked.
        """

        return np.ma.masked_where(self.data == 0, self.data)

    @lazyproperty
    def shape(self):
        """The shape of the segmentation array."""

        return self._data.shape

    @lazyproperty
    def _ndim(self):
        """The number of array dimensions of the segmentation array."""

        return self._data.ndim

    @lazyproperty
    def labels(self):
        """The sorted non-zero labels in the segmentation array."""

        return self._get_labels(self.data)

    @lazyproperty
    def nlabels(self):
        """The number of non-zero labels in the segmentation array."""

        return len(self.labels)

    @lazyproperty
    def max_label(self):
        import numpy as np
        """The maximum non-zero label in the segmentation array."""

        return np.max(self.labels)

    def get_index(self, label):
        """
        Find the index of the input ``label``.
        Parameters
        ----------
        labels : int
            The label numbers to find.
        Returns
        -------
        index : int
            The array index.
        Raises
        ------
        ValueError
            If ``label`` is invalid.
        """

        self.check_labels(label)
        return np.searchsorted(self.labels, label)

    def get_indices(self, labels):
        import numpy as np
        """
        Find the indices of the input ``labels``.
        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label numbers(s) to find.
        Returns
        -------
        indices : int `~numpy.ndarray`
            An integer array of indices with the same shape as
            ``labels``.  If ``labels`` is a scalar, then the returned
            index will also be a scalar.
        Raises
        ------
        ValueError
            If any input ``labels`` are invalid.
        """

        self.check_labels(labels)
        return np.searchsorted(self.labels, labels)

    @lazyproperty
    def slices(self):
        """
        A list of tuples, where each tuple contains two slices
        representing the minimal box that contains the labeled region.
        The list starts with the *non-zero* label.  The returned list
        has a length equal to the number of labels and matches the order
        of the ``labels`` attribute.
        """

        from scipy.ndimage import find_objects

        return [slc for slc in find_objects(self._data) if slc is not None]

    @lazyproperty
    def background_area(self):
        """The area (in pixel**2) of the background (label=0) region."""

        return len(self.data[self.data == 0])

    @lazyproperty
    def areas(self):
        import numpy as np
        """
        A 1D array of areas (in pixel**2) of the non-zero labeled
        regions.
        The `~numpy.ndarray` starts with the *non-zero* label.  The
        returned array has a length equal to the number of labels and
        matches the order of the ``labels`` attribute.
        """

        return np.array([area
                         for area in np.bincount(self.data.ravel())[1:]
                         if area != 0])

    def get_area(self, label):
        """
        The area (in pixel**2) of the region for the input label.
        Parameters
        ----------
        label : int
            The label whose area to return.  Label must be non-zero.
        Returns
        -------
        area : `~numpy.ndarray`
            The area of the labeled region.
        """

        return self.get_areas(label)

    def get_areas(self, labels):
        """
        The areas (in pixel**2) of the regions for the input labels.
        Parameters
        ----------
        labels : int, 1D array-like (int)
            The label(s) for which to return areas.  Label must be
            non-zero.
        Returns
        -------
        areas : `~numpy.ndarray`
            The areas of the labeled regions.
        """

        idx = self.get_indices(labels)
        return self.areas[idx]

    @lazyproperty
    def is_consecutive(self):
        """
        Determine whether or not the non-zero labels in the segmentation
        array are consecutive and start from 1.
        """

        return ((self.labels[-1] - self.labels[0] + 1) == self.nlabels and
                self.labels[0] == 1)

    @lazyproperty
    def missing_labels(self):
        """
        A 1D `~numpy.ndarray` of the sorted non-zero labels that are
        missing in the consecutive sequence from one to the maximum
        label number.
        """

        return np.array(sorted(set(range(0, self.max_label + 1))
                               .difference(np.insert(self.labels, 0, 0))))

    def copy(self):
        """Return a deep copy of this class instance."""

        return deepcopy(self)

    def check_label(self, label):
        """
        Check that the input label is a valid label number within the
        segmentation array.
        Parameters
        ----------
        label : int
            The label number to check.
        Raises
        ------
        ValueError
            If the input ``label`` is invalid.
        """

        self.check_labels(label)

    def check_labels(self, labels):
        """
        Check that the input label(s) are valid label numbers within the
        segmentation array.
        Parameters
        ----------
        labels : int, 1D array-like (int)
            The label(s) to check.
        Raises
        ------
        ValueError
            If any input ``labels`` are invalid.
        """

        labels = np.atleast_1d(labels)
        bad_labels = set()

        # check for positive label numbers
        idx = np.where(labels <= 0)[0]
        if idx.size > 0:
            bad_labels.update(labels[idx])

        # check if label is in the segmentation array
        bad_labels.update(np.setdiff1d(labels, self.labels))

        if bad_labels:
            if len(bad_labels) == 1:
                raise ValueError('label {} is invalid'.format(bad_labels))
            else:
                raise ValueError('labels {} are invalid'.format(bad_labels))

    @deprecated('0.7', alternative='make_cmap')
    def cmap(self, background_color='#000000', random_state=None):
        """
        Define a matplotlib colormap consisting of (random) muted
        colors.
        This is very useful for plotting the segmentation array.
        Parameters
        ----------
        background_color : str or `None`, optional
            A hex string in the "#rrggbb" format defining the first
            color in the colormap.  This color will be used as the
            background color (label = 0) when plotting the segmentation
            array.  The default is black ('#000000').
        random_state : int or `~numpy.random.mtrand.RandomState`, optional
            The pseudo-random number generator state used for random
            sampling.  Separate function calls with the same
            ``random_state`` will generate the same colormap.
        """

        return self.make_cmap(background_color=background_color,
                              random_state=random_state)  # pragma: no cover

    def make_cmap(self, background_color='#000000', random_state=None):
        """
        Define a matplotlib colormap consisting of (random) muted
        colors.
        This is very useful for plotting the segmentation array.
        Parameters
        ----------
        background_color : str or `None`, optional
            A hex string in the "#rrggbb" format defining the first
            color in the colormap.  This color will be used as the
            background color (label = 0) when plotting the segmentation
            array.  The default is black ('#000000').
        random_state : int or `~numpy.random.mtrand.RandomState`, optional
            The pseudo-random number generator state used for random
            sampling.  Separate function calls with the same
            ``random_state`` will generate the same colormap.
        Returns
        -------
        cmap : `matplotlib.colors.ListedColormap`
            The matplotlib colormap.
        """

        from matplotlib import colors

        cmap = make_random_cmap(self.max_label + 1, random_state=random_state)

        if background_color is not None:
            cmap.colors[0] = colors.hex2color(background_color)

        return cmap

    @deprecated('0.7', alternative='reassign_labels')
    def relabel(self, labels, new_label):
        """
        Reassign one or more label numbers.
        Multiple input ``labels`` will all be reassigned to the same
        ``new_label`` number.
        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label numbers(s) to reassign.
        new_label : int
            The reassigned label number.
        """

        self.reassign_label(labels, new_label)  # pragma: no cover

    def reassign_label(self, label, new_label, relabel=False):
        """
        Reassign a label number to a new number.
        If ``new_label`` is already present in the segmentation array,
        then it will be combined with the input ``label`` number.
        Parameters
        ----------
        labels : int
            The label number to reassign.
        new_label : int
            The newly assigned label number.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_label(label=1, new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_label(label=1, new_label=4)
        >>> segm.data
        array([[4, 4, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_label(label=1, new_label=4, relabel=True)
        >>> segm.data
        array([[2, 2, 0, 0, 2, 2],
               [0, 0, 0, 0, 0, 2],
               [0, 0, 1, 1, 0, 0],
               [4, 0, 0, 0, 0, 3],
               [4, 4, 0, 3, 3, 3],
               [4, 4, 0, 0, 3, 3]])
        """

        self.reassign_labels(label, new_label, relabel=relabel)

    def reassign_labels(self, labels, new_label, relabel=False):
        """
        Reassign one or more label numbers.
        Multiple input ``labels`` will all be reassigned to the same
        ``new_label`` number.  If ``new_label`` is already present in
        the segmentation array, then it will be combined with the input
        ``labels``.
        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label numbers(s) to reassign.
        new_label : int
            The reassigned label number.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_labels(labels=[1, 7], new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [2, 0, 0, 0, 0, 5],
               [2, 2, 0, 5, 5, 5],
               [2, 2, 0, 0, 5, 5]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_labels(labels=[1, 7], new_label=4)
        >>> segm.data
        array([[4, 4, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [4, 0, 0, 0, 0, 5],
               [4, 4, 0, 5, 5, 5],
               [4, 4, 0, 0, 5, 5]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_labels(labels=[1, 7], new_label=2, relabel=True)
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [1, 0, 0, 0, 0, 4],
               [1, 1, 0, 4, 4, 4],
               [1, 1, 0, 0, 4, 4]])
        """

        self.check_labels(labels)

        labels = np.atleast_1d(labels)
        if labels.size == 0:
            return

        idx = np.zeros(self.max_label + 1, dtype=int)
        idx[self.labels] = self.labels
        idx[labels] = new_label  # reassign labels

        if relabel:
            labels = np.unique(idx[idx != 0])
            idx2 = np.zeros(max(labels) + 1, dtype=np.int)
            idx2[labels] = np.arange(len(labels)) + 1
            idx = idx2[idx]

        data_new = idx[self.data]
        self.__dict__ = {}  # reset all cached properties
        self._data = data_new  # use _data to avoid validation

    def relabel_consecutive(self, start_label=1):
        """
        Reassign the label numbers consecutively starting from a given
        label number.
        Parameters
        ----------
        start_label : int, optional
            The starting label number, which should be a strictly
            positive integer.  The default is 1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.relabel_consecutive()
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [5, 0, 0, 0, 0, 4],
               [5, 5, 0, 4, 4, 4],
               [5, 5, 0, 0, 4, 4]])
        """

        if start_label <= 0:
            raise ValueError('start_label must be > 0.')

        if ((self.labels[0] == start_label) and
                (self.labels[-1] - self.labels[0] + 1) == self.nlabels):
            return

        new_labels = np.zeros(self.max_label + 1, dtype=np.int)
        new_labels[self.labels] = np.arange(self.nlabels) + start_label

        data_new = new_labels[self.data]
        self.__dict__ = {}  # reset all cached properties
        self._data = data_new  # use _data to avoid validation

    def keep_label(self, label, relabel=False):
        """
        Keep only the specified label.
        Parameters
        ----------
        label : int
            The label number to keep.
        relabel : bool, optional
            If `True`, then the single segment will be assigned a label
            value of 1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_label(label=3)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_label(label=3, relabel=True)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
        """

        self.keep_labels(label, relabel=relabel)

    def keep_labels(self, labels, relabel=False):
        """
        Keep only the specified labels.
        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label number(s) to keep.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_labels(labels=[5, 3])
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 5],
               [0, 0, 0, 5, 5, 5],
               [0, 0, 0, 0, 5, 5]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_labels(labels=[5, 3], relabel=True)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 2],
               [0, 0, 0, 2, 2, 2],
               [0, 0, 0, 0, 2, 2]])
        """

        self.check_labels(labels)

        labels = np.atleast_1d(labels)
        labels_tmp = list(set(self.labels) - set(labels))
        self.remove_labels(labels_tmp, relabel=relabel)

    def remove_label(self, label, relabel=False):
        """
        Remove the label number.
        The removed label is assigned a value of zero (i.e.,
        background).
        Parameters
        ----------
        label : int
            The label number to remove.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_label(label=5)
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_label(label=5, relabel=True)
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [4, 0, 0, 0, 0, 0],
               [4, 4, 0, 0, 0, 0],
               [4, 4, 0, 0, 0, 0]])
        """

        self.remove_labels(label, relabel=relabel)

    def remove_labels(self, labels, relabel=False):
        """
        Remove one or more labels.
        Removed labels are assigned a value of zero (i.e., background).
        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label number(s) to remove.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_labels(labels=[5, 3])
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_labels(labels=[5, 3], relabel=True)
        >>> segm.data
        array([[1, 1, 0, 0, 2, 2],
               [0, 0, 0, 0, 0, 2],
               [0, 0, 0, 0, 0, 0],
               [3, 0, 0, 0, 0, 0],
               [3, 3, 0, 0, 0, 0],
               [3, 3, 0, 0, 0, 0]])
        """

        self.check_labels(labels)
        self.reassign_labels(labels, new_label=0, relabel=relabel)

    def remove_border_labels(self, border_width, partial_overlap=True,
                             relabel=False):
        """
        Remove labeled segments near the array border.
        Labels within the defined border region will be removed.
        Parameters
        ----------
        border_width : int
            The width of the border region in pixels.
        partial_overlap : bool, optional
            If this is set to `True` (the default), a segment that
            partially extends into the border region will be removed.
            Segments that are completely within the border region are
            always removed.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_border_labels(border_width=1)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_border_labels(border_width=1,
        ...                           partial_overlap=False)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        """

        if border_width >= min(self.shape) / 2:
            raise ValueError('border_width must be smaller than half the '
                             'array size in any dimension')

        border_mask = np.zeros(self.shape, dtype=bool)
        for i in range(border_mask.ndim):
            border_mask = border_mask.swapaxes(0, i)
            border_mask[:border_width] = True
            border_mask[-border_width:] = True
            border_mask = border_mask.swapaxes(0, i)

        self.remove_masked_labels(border_mask,
                                  partial_overlap=partial_overlap,
                                  relabel=relabel)

    def remove_masked_labels(self, mask, partial_overlap=True,
                             relabel=False):
        """
        Remove labeled segments located within a masked region.
        Parameters
        ----------
        mask : array_like (bool)
            A boolean mask, with the same shape as the segmentation
            array, where `True` values indicate masked pixels.
        partial_overlap : bool, optional
            If this is set to `True` (default), a segment that partially
            extends into a masked region will also be removed.  Segments
            that are completely within a masked region are always
            removed.
        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> mask = np.zeros(segm.data.shape, dtype=bool)
        >>> mask[0, :] = True  # mask the first row
        >>> segm.remove_masked_labels(mask)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_masked_labels(mask, partial_overlap=False)
        >>> segm.data
        array([[0, 0, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        """

        if mask.shape != self.shape:
            raise ValueError('mask must have the same shape as the '
                             'segmentation array')
        remove_labels = self._get_labels(self.data[mask])
        if not partial_overlap:
            interior_labels = self._get_labels(self.data[~mask])
            remove_labels = list(set(remove_labels) - set(interior_labels))
        self.remove_labels(remove_labels, relabel=relabel)

    def outline_segments(self, mask_background=False):
        """
        Outline the labeled segments.
        The "outlines" represent the pixels *just inside* the segments,
        leaving the background pixels unmodified.
        Parameters
        ----------
        mask_background : bool, optional
            Set to `True` to mask the background pixels (labels = 0) in
            the returned array.  This is useful for overplotting the
            segment outlines.  The default is `False`.
        Returns
        -------
        boundaries : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            An array with the same shape of the segmentation array
            containing only the outlines of the labeled segments.  The
            pixel values in the outlines correspond to the labels in the
            segmentation array.  If ``mask_background`` is `True`, then
            a `~numpy.ma.MaskedArray` is returned.
        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[0, 0, 0, 0, 0, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 0, 0, 0, 0, 0]])
        >>> segm.outline_segments()
        array([[0, 0, 0, 0, 0, 0],
               [0, 2, 2, 2, 2, 0],
               [0, 2, 0, 0, 2, 0],
               [0, 2, 0, 0, 2, 0],
               [0, 2, 2, 2, 2, 0],
               [0, 0, 0, 0, 0, 0]])
        """

        from scipy.ndimage import (generate_binary_structure, grey_dilation,
                                   grey_erosion)

        # mode='constant' ensures outline is included on the array borders
        selem = generate_binary_structure(self._ndim, 1)  # edge connectivity
        eroded = grey_erosion(self.data, footprint=selem, mode='constant',
                              cval=0.)
        dilated = grey_dilation(self.data, footprint=selem, mode='constant',
                                cval=0.)

        outlines = ((dilated != eroded) & (self.data != 0)).astype(int)
        outlines *= self.data

        if mask_background:
            outlines = np.ma.masked_where(outlines == 0, outlines)

        return outlines



import astropy
from astropy.version import version as astropy_version
from astropy.units import Quantity


def _make_binary_structure(ndim, connectivity):
    """
    Make a binary structure element.
    Parameters
    ----------
    ndim : int
        The number of array dimensions.
    connectivity : {4, 8}
        For the case of ``ndim=2``, the type of pixel connectivity used
        in determining how pixels are grouped into a detected source.
        The options are 4 or 8 (default).  4-connected pixels touch
        along their edges.  8-connected pixels touch along their edges
        or corners.  For reference, SExtractor uses 8-connected pixels.
    Returns
    -------
    array : ndarray of int or bool
        The binary structure element.  If ``ndim <= 2`` an array of int
        is returned, otherwise an array of bool is returned.
    """

    from scipy.ndimage import generate_binary_structure
    import numpy as np

    if ndim == 1:
        selem = np.array((1, 1, 1))
    elif ndim == 2:
        if connectivity == 4:
            selem = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
        elif connectivity == 8:
            selem = np.ones((3, 3), dtype=int)
        else:
            raise ValueError('Invalid connectivity={0}.  '
                             'Options are 4 or 8'.format(connectivity))
    else:
        selem = generate_binary_structure(ndim, 1)

    return selem


from astropy.stats import sigma_clipped_stats

def detect_threshold(data, nsigma, background=None, error=None, mask=None,
                     mask_value=None, sigclip_sigma=3.0, sigclip_iters=None):
    """
    Calculate a pixel-wise threshold image that can be used to detect
    sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    nsigma : float
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.

    background : float or array_like, optional
        The background value(s) of the input ``data``.  ``background``
        may either be a scalar value or a 2D image with the same shape
        as the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to ``0.0``.  If
        `None`, then a scalar background value will be estimated using
        sigma-clipped statistics.

    error : float or array_like, optional
        The Gaussian 1-sigma standard deviation of the background noise
        in ``data``.  ``error`` should include all sources of
        "background" error, but *exclude* the Poisson error of the
        sources.  If ``error`` is a 2D image, then it should represent
        the 1-sigma background error in each pixel of ``data``.  If
        `None`, then a scalar background rms value will be estimated
        using sigma-clipped statistics.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the image background
        statistics.

    mask_value : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image background statistics.  ``mask_value`` will
        be ignored if ``mask`` is input.

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    sigclip_iters : int, optional
       The number of iterations to perform sigma clipping, or `None` to
       clip until convergence is achieved (i.e., continue until the last
       iteration clips nothing) when calculating the image background
       statistics.

    Returns
    -------
    threshold : 2D `~numpy.ndarray`
        A 2D image with the same shape as ``data`` containing the
        pixel-wise threshold values.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`

    Notes
    -----
    The ``mask``, ``mask_value``, ``sigclip_sigma``, and
    ``sigclip_iters`` inputs are used only if it is necessary to
    estimate ``background`` or ``error`` using sigma-clipped background
    statistics.  If ``background`` and ``error`` are both input, then
    ``mask``, ``mask_value``, ``sigclip_sigma``, and ``sigclip_iters``
    are ignored.
    """
    ###THESE IMPORTS ARE NOT HERE IN THE ORIGINAL###
    import astropy
    from astropy.version import version as astropy_version
    from astropy.units import Quantity
    from astropy.stats import sigma_clipped_stats
    from astropy.convolution import Kernel2D
    import numpy as np
    if background is None or error is None:
        if astropy_version < '3.1':
            data_mean, _, data_std = sigma_clipped_stats(
                data, mask=mask, mask_value=mask_value, sigma=sigclip_sigma,
                iters=sigclip_iters)
        else:
            data_mean, _, data_std = sigma_clipped_stats(
                data, mask=mask, mask_value=mask_value, sigma=sigclip_sigma,
                maxiters=sigclip_iters)

        bkgrd_image = np.zeros_like(data) + data_mean
        bkgrdrms_image = np.zeros_like(data) + data_std

    if background is None:
        background = bkgrd_image
    else:
        if np.isscalar(background):
            background = np.zeros_like(data) + background
        else:
            if background.shape != data.shape:
                raise ValueError('If input background is 2D, then it '
                                 'must have the same shape as the input '
                                 'data.')

    if error is None:
        error = bkgrdrms_image
    else:
        if np.isscalar(error):
            error = np.zeros_like(data) + error
        else:
            if error.shape != data.shape:
                raise ValueError('If input error is 2D, then it '
                                 'must have the same shape as the input '
                                 'data.')

    return background + (error * nsigma)

def _detect_sources(data, thresholds, npixels, filter_kernel=None,
                    connectivity=8, mask=None, deblend_skip=False):
    """
    Detect sources above a specified threshold value in an image and
    return a `~photutils.segmentation.SegmentationImage` object.
    Detected sources must have ``npixels`` connected pixels that are
    each greater than the ``threshold`` value.  If the filtering option
    is used, then the ``threshold`` is applied to the filtered image.
    The input ``mask`` can be used to mask pixels in the input data.
    Masked pixels will not be included in any source.
    This function does not deblend overlapping sources.  First use this
    function to detect sources followed by
    :func:`~photutils.segmentation.deblend_sources` to deblend sources.
    Parameters
    ----------
    data : array_like
        The 2D array of the image.
    thresholds : array-like of floats or arrays
        The data value or pixel-wise data values to be used for the
        detection thresholds.  A 2D ``threshold`` must have the same
        shape as ``data``.  See `~photutils.detection.detect_threshold`
        for one way to create a ``threshold`` image.
    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.
    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.
    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 4 or 8
        (default).  4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.  For
        reference, SExtractor uses 8-connected pixels.
    mask : array_like of bool, optional
        A boolean mask, with the same shape as the input ``data``, where
        `True` values indicate masked pixels.  Masked pixels will not be
        included in any source.
    deblend_skip : bool, optional
        If `True` do not include the segmentation image in the output
        list for any threshold level where the number of detected
        sources is less than 2.  This is useful for source deblending
        and improves its performance.
    Returns
    -------
    segment_image : list of `~photutils.segmentation.SegmentationImage`
        A list of 2D segmentation images, with the same shape as
        ``data``, where sources are marked by different positive integer
        values.  A value of zero is reserved for the background.  If no
        sources are found for a given threshold, then the output list
        will contain `None` for that threshold.  Also see the
        ``deblend_skip`` keyword.
    """

    from scipy import ndimage
    import numpy as np

    if (npixels <= 0) or (int(npixels) != npixels):
        raise ValueError('npixels must be a positive integer, got '
                         '"{0}"'.format(npixels))

    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError('mask must have the same shape as the input '
                             'image.')

    if filter_kernel is not None:
        data = _filter_data(data, filter_kernel, mode='constant',
                            fill_value=0.0, check_normalization=True)

    # ignore RuntimeWarning caused by > comparison when data contains NaNs
    import warnings
    warnings.simplefilter('ignore', category=RuntimeWarning)

    selem = _make_binary_structure(data.ndim, connectivity)

    segms = []
    for threshold in thresholds:
        data2 = data > threshold

        if mask is not None:
            data2 &= ~mask

        # return if threshold was too high to detect any sources
        if np.count_nonzero(data2) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            if deblend_skip:
                continue
            else:
                segms.append(None)
                continue

        segm_img, _ = ndimage.label(data2, structure=selem)

        # remove objects with less than npixels
        # NOTE:  for typical data, making the cutout images is ~10x faster
        # than using segm_img directly
        segm_slices = ndimage.find_objects(segm_img)
        for i, slices in enumerate(segm_slices):
            cutout = segm_img[slices]
            segment_mask = (cutout == (i+1))
            if np.count_nonzero(segment_mask) < npixels:
                cutout[segment_mask] = 0

        if np.count_nonzero(segm_img) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            if deblend_skip:
                continue
            else:
                segms.append(None)
                continue

        segm = object.__new__(SegmentationImage)
        segm._data = segm_img

        if deblend_skip and segm.nlabels == 1:
            continue
        else:
            segm.relabel_consecutive()
            segms.append(segm)

    return segms


def _filter_data(data, kernel, mode='constant', fill_value=0.0,
                 check_normalization=False):
    """
    Convolve a 2D image with a 2D kernel.
    The kernel may either be a 2D `~numpy.ndarray` or a
    `~astropy.convolution.Kernel2D` object.
    Parameters
    ----------
    data : array_like
        The 2D array of the image.
    kernel : array-like (2D) or `~astropy.convolution.Kernel2D`
        The 2D kernel used to filter the input ``data``. Filtering the
        ``data`` will smooth the noise and maximize detectability of
        objects with a shape similar to the kernel.
    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` determines how the array borders are handled.  For
        the ``'constant'`` mode, values outside the array borders are
        set to ``fill_value``.  The default is ``'constant'``.
    fill_value : scalar, optional
        Value to fill data values beyond the array borders if ``mode``
        is ``'constant'``.  The default is ``0.0``.
    check_normalization : bool, optional
        If `True` then a warning will be issued if the kernel is not
        normalized to 1.
    """

    from scipy import ndimage
    import numpy as np

    if kernel is not None:
        if isinstance(kernel, Kernel2D):
            kernel_array = kernel.array
        else:
            kernel_array = kernel

        if check_normalization:
            if not np.allclose(np.sum(kernel_array), 1.0):
                warnings.warn('The kernel is not normalized.',
                              AstropyUserWarning)

        # scipy.ndimage.convolve currently strips units, but be explicit
        # in case that behavior changes
        unit = None
        if isinstance(data, Quantity):
            unit = data.unit
            data = data.value

        # NOTE:  astropy.convolution.convolve fails with zero-sum
        # kernels (used in findstars) (cf. astropy #1647)
        # NOTE: if data is int and kernel is float, ndimage.convolve
        # will return an int image - here we make the data float so
        # that a float image is always returned
        result = ndimage.convolve(data.astype(float), kernel_array,
                                  mode=mode, cval=fill_value)

        if unit is not None:
            result = result * unit  # can't use *= with older astropy

        return result
    else:
        return data


from astropy.convolution import Kernel2D


def detect_sources(data, threshold, npixels, filter_kernel=None,
                   connectivity=8, mask=None):
    import numpy as np
    """
    Detect sources above a specified threshold value in an image and
    return a `~photutils.segmentation.SegmentationImage` object.

    Detected sources must have ``npixels`` connected pixels that are
    each greater than the ``threshold`` value.  If the filtering option
    is used, then the ``threshold`` is applied to the filtered image.
    The input ``mask`` can be used to mask pixels in the input data.
    Masked pixels will not be included in any source.

    This function does not deblend overlapping sources.  First use this
    function to detect sources followed by
    :func:`~photutils.segmentation.deblend_sources` to deblend sources.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same
        shape as ``data``.  See `~photutils.detection.detect_threshold`
        for one way to create a ``threshold`` image.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 4 or 8
        (default).  4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.  For
        reference, SExtractor uses 8-connected pixels.

    mask : array_like of bool, optional
        A boolean mask, with the same shape as the input ``data``, where
        `True` values indicate masked pixels.  Masked pixels will not be
        included in any source.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  If no sources
        are found then `None` is returned.

    See Also
    --------
    :func:`photutils.detection.detect_threshold`,
    :class:`photutils.segmentation.SegmentationImage`,
    :func:`photutils.segmentation.source_properties`
    :func:`photutils.segmentation.deblend_sources`

    Examples
    --------

    .. plot::
        :include-source:

        # make a table of Gaussian sources
        from astropy.table import Table
        table = Table()
        table['amplitude'] = [50, 70, 150, 210]
        table['x_mean'] = [160, 25, 150, 90]
        table['y_mean'] = [70, 40, 25, 60]
        table['x_stddev'] = [15.2, 5.1, 3., 8.1]
        table['y_stddev'] = [2.6, 2.5, 3., 4.7]
        table['theta'] = np.array([145., 20., 0., 60.]) * np.pi / 180.

        # make an image of the sources with Gaussian noise
        from photutils.datasets import make_gaussian_sources_image
        from photutils.datasets import make_noise_image
        shape = (100, 200)
        sources = make_gaussian_sources_image(shape, table)
        noise = make_noise_image(shape, distribution='gaussian', mean=0.,
                                 stddev=5., random_state=12345)
        image = sources + noise

        # detect the sources
        from photutils import detect_threshold, detect_sources
        threshold = detect_threshold(image, nsigma=3)
        from astropy.convolution import Gaussian2DKernel
        kernel_sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM = 3
        kernel = Gaussian2DKernel(kernel_sigma, x_size=3, y_size=3)
        kernel.normalize()
        segm = detect_sources(image, threshold, npixels=5,
                              filter_kernel=kernel)

        # plot the image and the segmentation image
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.imshow(image, origin='lower', interpolation='nearest')
        ax2.imshow(segm.data, origin='lower', interpolation='nearest')
    """

    return _detect_sources(data, (threshold,), npixels,
                           filter_kernel=filter_kernel,
                           connectivity=connectivity, mask=mask)[0]

def deblend_sources(data, segment_img, npixels, filter_kernel=None,
                    labels=None, nlevels=32, contrast=0.001,
                    mode='exponential', connectivity=8, relabel=True):
    import numpy as np
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_.  In
    order to deblend sources, there must be a saddle between them.

    Parameters
    ----------
    data : array_like
        The data array.

    segment_img : `~photutils.segmentation.SegmentationImage` or array_like (int)
        A segmentation image, either as a
        `~photutils.segmentation.SegmentationImage` object or an
        `~numpy.ndarray`, with the same shape as ``data`` where sources
        are labeled by different positive integer values.  A value of
        zero is reserved for the background.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    filter_kernel : array-like or `~astropy.convolution.Kernel2D`, optional
        The array of the kernel used to filter the image before
        thresholding.  Filtering the image will smooth the noise and
        maximize detectability of objects with a shape similar to the
        kernel.

    labels : int or array-like of int, optional
        The label numbers to deblend.  If `None` (default), then all
        labels in the segmentation image will be deblended.

    nlevels : int, optional
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object.  ``contrast`` must be between 0
        and 1, inclusive.  If ``contrast = 0`` then every local peak
        will be made a separate object (maximum deblending).  If
        ``contrast = 1`` then no deblending will occur.  The default is
        0.001, which will deblend sources with a 7.5 magnitude
        difference.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).  The
        default is 'exponential'.

    connectivity : {8, 4}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source.  The options are 8 (default)
        or 4.  8-connected pixels touch along their edges or corners.
        4-connected pixels touch along their edges.  For reference,
        SExtractor uses 8-connected pixels.

    relabel : bool
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in consecutive order starting
        from 1.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.detect_sources`
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if labels is None:
        labels = segment_img.labels
    labels = np.atleast_1d(labels)
    segment_img.check_labels(labels)

    if filter_kernel is not None:
        data = _filter_data(data, filter_kernel, mode='constant',
                            fill_value=0.0)

    last_label = segment_img.max_label
    segm_deblended = object.__new__(SegmentationImage)
    segm_deblended._data = np.copy(segment_img.data)
    for label in labels:
        source_slice = segment_img.slices[segment_img.get_index(label)]
        source_data = data[source_slice]

        source_segm = object.__new__(SegmentationImage)
        source_segm._data = np.copy(segment_img.data[source_slice])

        source_segm.keep_labels(label)  # include only one label
        source_deblended = _deblend_source(
            source_data, source_segm, npixels, nlevels=nlevels,
            contrast=contrast, mode=mode, connectivity=connectivity)

        if not np.array_equal(source_deblended.data.astype(bool),
                              source_segm.data.astype(bool)):
            raise ValueError('Deblending failed for source "{0}".  Please '
                             'ensure you used the same pixel connectivity '
                             'in detect_sources and deblend_sources.  If '
                             'this issue persists, then please inform the '
                             'developers.'.format(label))

        if source_deblended.nlabels > 1:
            # replace the original source with the deblended source
            source_mask = (source_deblended.data > 0)
            segm_tmp = segm_deblended.data
            segm_tmp[source_slice][source_mask] = (
                source_deblended.data[source_mask] + last_label)

            segm_deblended.__dict__ = {}  # reset cached properties
            segm_deblended._data = segm_tmp

            last_label += source_deblended.nlabels

    if relabel:
        segm_deblended.relabel_consecutive()

    return segm_deblended
