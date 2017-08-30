#include "WFF.h"
#include "mem_manager.h"

/* Single precision Data structures for WFT Results */
WFT2_DeviceResultsF::WFT2_DeviceResultsF()
	: m_d_filtered(nullptr)
	, m_d_wx(nullptr)
	, m_d_wy(nullptr)
	, m_d_phase(nullptr)
	, m_d_phase_comp(nullptr)
	, m_d_b(nullptr)
	, m_d_r(nullptr)
	, m_d_cxx(nullptr)
	, m_d_cyy(nullptr)
{}

WFT2_DeviceResultsF::~WFT2_DeviceResultsF()
{
	cudaSafeFree(m_d_filtered);

	cudaSafeFree(m_d_wx);
	cudaSafeFree(m_d_phase);
	cudaSafeFree(m_d_phase_comp);
	cudaSafeFree(m_d_b);
	cudaSafeFree(m_d_r);
	cudaSafeFree(m_d_cxx);
	cudaSafeFree(m_d_cyy);
}

int getFirstGreater(int x)
{
	int first = 0, last = int(OPT_FFT_SIZE.size()) - 1;
	while (first <= last)
	{
		int mid = (first + last) / 2;

		if (OPT_FFT_SIZE[mid] >= x)
			last = mid - 1;
		else
			first = mid + 1;
	}
	return last + 1 == OPT_FFT_SIZE.size() ? -1 : last + 1;
}