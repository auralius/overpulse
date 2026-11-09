#include "Noodle2.h"
#if defined(ARDUINO_ARCH_ESP8266)
#include <Arduino.h>
#endif
#include <string.h>
#include <stdlib.h>
#include "Noodle2KernelInternal.h"
#ifdef NOODLE2_TRACE_CONV
#include <stdio.h>
#endif

namespace noodle2 {

struct Geometry {
    uint16_t conv_h, conv_w, out_h, out_w;
    uint8_t pool_kh, pool_kw, pool_sh, pool_sw;
};
// Resolve dynamic SAME/VALID padding once, then use these explicit offsets in
// conv_at(). No temporary padded input tensor is allocated.
static Status geometry(uint16_t h, uint16_t w, Conv2D& k, Pool p, Geometry& g) {
    if (!h || !w || !k.kh || !k.kw || !k.sh || !k.sw) return BAD_SHAPE;
    if (k.padding == VALID_PADDING) {
        k.pt = k.pb = k.pl = k.pr = 0;
    } else if (k.padding == SAME_PADDING) {
        const uint32_t oh = (uint32_t(h) + k.sh - 1U) / k.sh;
        const uint32_t ow = (uint32_t(w) + k.sw - 1U) / k.sw;
        const uint32_t needed_h = (oh - 1U) * k.sh + k.kh;
        const uint32_t needed_w = (ow - 1U) * k.sw + k.kw;
        const uint32_t total_h = needed_h > h ? needed_h - h : 0U;
        const uint32_t total_w = needed_w > w ? needed_w - w : 0U;
        k.pt = static_cast<uint8_t>(total_h / 2U);
        k.pb = static_cast<uint8_t>(total_h - k.pt);
        k.pl = static_cast<uint8_t>(total_w / 2U);
        k.pr = static_cast<uint8_t>(total_w - k.pl);
    } else if (k.padding != EXPLICIT_PADDING) {
        return UNSUPPORTED;
    }
    int32_t padded_h = int32_t(h)+k.pt+k.pb;
    int32_t padded_w = int32_t(w)+k.pl+k.pr;
    if (padded_h < k.kh || padded_w < k.kw) return BAD_SHAPE;
    uint32_t oh = uint32_t(padded_h - k.kh)/k.sh + 1;
    uint32_t ow = uint32_t(padded_w - k.kw)/k.sw + 1;
    if (oh > UINT16_MAX || ow > UINT16_MAX) return BAD_SHAPE;
    g.conv_h=static_cast<uint16_t>(oh); g.conv_w=static_cast<uint16_t>(ow);
    if (p.type == NO_POOL) {
        g.pool_kh=g.pool_kw=g.pool_sh=g.pool_sw=1;
    } else {
        if (p.type != MAX_POOL && p.type != AVG_POOL) return UNSUPPORTED;
        if (!p.kh || !p.kw || !p.sh || !p.sw || oh < p.kh || ow < p.kw)
            return BAD_SHAPE;
        g.pool_kh=p.kh; g.pool_kw=p.kw; g.pool_sh=p.sh; g.pool_sw=p.sw;
    }
    uint32_t ph = (oh-g.pool_kh)/g.pool_sh + 1;
    uint32_t pw = (ow-g.pool_kw)/g.pool_sw + 1;
    if (!ph || !pw || ph>UINT16_MAX || pw>UINT16_MAX) return BAD_SHAPE;
    g.out_h=static_cast<uint16_t>(ph); g.out_w=static_cast<uint16_t>(pw);
    return OK;
}
Status conv2d_output_shape(uint16_t in_h, uint16_t in_w, Conv2D conv,
                           Pool pool, uint16_t& out_h, uint16_t& out_w) {
    Geometry result;
    const Status st = geometry(in_h, in_w, conv, pool, result);
    if (st == OK) { out_h = result.out_h; out_w = result.out_w; }
    return st;
}
static int8_t rounded_average(int32_t sum, uint32_t count) {
    int64_t a = sum;
    bool negative = a < 0;
    uint64_t mag = negative ? uint64_t(-a) : uint64_t(a);
    uint64_t n = (mag + count/2)/count;
    int32_t rounded = negative ? -static_cast<int32_t>(n) : static_cast<int32_t>(n);
    if (rounded < -128) rounded=-128;
    if (rounded > 127) rounded=127;
    return static_cast<int8_t>(rounded);
}
// One convolution result at a location; no intermediate activation buffers.
static Status conv_at(const NoodleTensor& in, const NoodleParams& p,
                      Conv2D k, uint16_t o, uint16_t y, uint16_t x,
                      bool depthwise, Activation act, int8_t& result) {
    Status st;
    int32_t acc;
    if ((st = p.bias(o, acc)) != OK) return st;
#ifdef NOODLE2_TRACE_CONV
    // Follow only Conv1, output channel 1, pool output at (5,5).
    // A 2x2 average pool samples convolution positions (10,10)..(11,11).
    const bool trace_region = in.channels()==1 && in.height()==28 &&
        in.width()==28 && o==1 && y>=10 && y<=11 && x>=10 && x<=11;
    const bool trace_mac = trace_region && y==10 && x==10;
    if (trace_region) printf("[CONV1] oc=%u y=%u x=%u initial_bias=%ld\n",
        unsigned(o), unsigned(y), unsigned(x), long(acc));
#endif
    const uint16_t ic_start = depthwise ? uint16_t(o / (p.outputs()/in.channels())) : 0;
    const uint16_t ic_end = depthwise ? uint16_t(ic_start + 1) : in.channels();
    for (uint16_t ic=ic_start; ic<ic_end; ++ic) {
        for (uint8_t ky=0; ky<k.kh; ++ky) {
            int32_t iy=int32_t(y)*k.sh+ky-k.pt;
            if (iy<0 || iy>=in.height()) continue;
            for (uint8_t kx=0; kx<k.kw; ++kx) {
                int32_t ix=int32_t(x)*k.sw+kx-k.pl;
                if (ix<0 || ix>=in.width()) continue;
                uint32_t inputidx=(uint32_t(ic)*in.height()+uint32_t(iy))*in.width()+uint32_t(ix);
                uint32_t wi = depthwise ? ((uint32_t(o)*k.kh+ky)*k.kw+kx)
                        : (((uint32_t(o)*in.channels()+ic)*k.kh+ky)*k.kw+kx);
                int8_t v, weight;
                if ((st = in.get(inputidx, v)) != OK ||
                    (st = p.weight(wi, weight)) != OK) return st;
#ifdef NOODLE2_TRACE_CONV
                const int32_t acc_before = acc;
#endif
                if (!safe_accumulate(acc, int32_t(v)*weight)) return OVERFLOW;
#ifdef NOODLE2_TRACE_CONV
                if (trace_mac) printf("  ic=%u ky=%u kx=%u inputidx=%lu wi=%lu v=%d w=%d product=%ld acc:%ld->%ld\n",
                    unsigned(ic), unsigned(ky), unsigned(kx),
                    static_cast<unsigned long>(inputidx), static_cast<unsigned long>(wi),
                    int(v), int(weight), long(int32_t(v)*weight),
                    long(acc_before), long(acc));
#endif
            }
        }
    }
    result=apply_activation(requantize(acc,p.quant()),act);
#ifdef NOODLE2_TRACE_CONV
    if (trace_region) printf("[CONV1] oc=%u y=%u x=%u acc=%ld quantized=%d activation=%d\n",
        unsigned(o), unsigned(y), unsigned(x), long(acc),
        int(requantize(acc,p.quant())), int(result));
#endif
    return OK;
}
static Status convolution(const NoodleTensor& in, NoodleTensor& out,
                          const NoodleParams& p, Conv2D k, Activation act,
                          Pool pool, bool depthwise) {
    if (!supports_fused_activation(act)) return UNSUPPORTED;
    if (&in==&out || &in.buffer()==&out.buffer() ||
        (in.buffer().address() && in.buffer().address()==out.buffer().address()))
        return ALIASED_TENSORS;
    if (!in.size()) return BAD_SHAPE;
    if (depthwise && (p.outputs() % in.channels())) return BAD_PARAMS;
    uint32_t expected;
    if (!size4_checked(depthwise ? 1 : in.channels(), p.outputs(), k.kh, k.kw, expected))
        return OVERFLOW;
    Status st = p.validate(expected, p.outputs());
    if (st != OK) return st;
    Geometry g;
    if ((st = geometry(in.height(), in.width(), k, pool, g)) != OK) return st;
    if ((st = out.resize(p.outputs(), g.out_h, g.out_w)) != OK) return st;
    for (uint16_t oc=0; oc<p.outputs(); ++oc) {
        for (uint16_t py=0; py<g.out_h; ++py) {
            for (uint16_t px=0; px<g.out_w; ++px) {
                int8_t maxval=-128;
                int32_t sum=0;
                uint16_t n=0;
                for (uint8_t wy=0; wy<g.pool_kh; ++wy) {
                    for (uint8_t wx=0; wx<g.pool_kw; ++wx) {
                        uint16_t y=uint16_t(uint32_t(py)*g.pool_sh+wy);
                        uint16_t x=uint16_t(uint32_t(px)*g.pool_sw+wx);
                        int8_t val;
                        if ((st=conv_at(in,p,k,oc,y,x,depthwise,act,val)) != OK) return st;
                        if (val>maxval) maxval=val;
                        sum+=val; ++n;
                    }
                }
                int8_t value = pool.type==MAX_POOL ? maxval :
                    (pool.type==AVG_POOL ? rounded_average(sum,n) : maxval);
                uint32_t dst=(uint32_t(oc)*g.out_h+py)*g.out_w+px;
#ifdef NOODLE2_TRACE_CONV
                if (in.channels()==1 && in.height()==28 && in.width()==28 &&
                    oc==1 && py==5 && px==5) {
                    printf("[AVGPOOL] oc=1 py=5 px=5 sum=%ld count=%u output=%d dst=%lu\n",
                        long(sum), unsigned(n), int(value),
                        static_cast<unsigned long>(dst));
                }
#endif
                if ((st=out.put(dst,value)) != OK) return st;
            }
        }
    }
    return OK;
}
// Tiled Conv2D: cache one CHW output-channel kernel, then traverse spatial
// tiles of the FINAL (possibly pooled) output. No full intermediate tensor.
// Requantize -> activate -> pool exactly as the direct reference kernel.
#ifndef NOODLE2_TILE_MAX_WEIGHTS
#define NOODLE2_TILE_MAX_WEIGHTS 256
#endif
Status conv2d_tiled(const NoodleTensor& in, NoodleTensor& out,
                    const NoodleParams& p, Conv2D k, Activation act, Pool pool,
                    uint8_t tile_h, uint8_t tile_w) {
    if (!tile_h || !tile_w) return BAD_PARAMS;
    if (!supports_fused_activation(act)) return UNSUPPORTED;
    if (&in == &out || &in.buffer() == &out.buffer() ||
        (in.buffer().address() && in.buffer().address()==out.buffer().address()))
        return ALIASED_TENSORS;
    const uint64_t weight_per_oc=uint64_t(in.channels())*k.kh*k.kw;
    if (!weight_per_oc || weight_per_oc>NOODLE2_TILE_MAX_WEIGHTS)
        return convolution(in,out,p,k,act,pool,false);
    uint32_t expected=0;
    if (!size4_checked(in.channels(),p.outputs(),k.kh,k.kw,expected)) return OVERFLOW;
    Status st=p.validate(expected,p.outputs());
    if (st!=OK) return st;
    Geometry g;
    if ((st=geometry(in.height(),in.width(),k,pool,g))!=OK) return st;
    if ((st=out.resize(p.outputs(),g.out_h,g.out_w))!=OK) return st;
    int8_t weights[NOODLE2_TILE_MAX_WEIGHTS];
    // AVR Flash is not directly dereferenceable as ordinary RAM.
#if defined(__AVR__)
    const uint8_t* in_data=0;
#else
    const uint8_t* in_data=in.buffer().address();
#endif
    for (uint16_t oc=0; oc<p.outputs(); ++oc) {
        int32_t bias=0;
        if ((st=p.bias(oc,bias))!=OK) return st;
        // A single forward pass through this channel's weights.
        for (uint32_t wi=0; wi<weight_per_oc; ++wi)
            if ((st=p.weight(uint32_t(oc)*uint32_t(weight_per_oc)+wi,weights[wi]))!=OK)
                return st;
        for (uint32_t ty=0; ty<g.out_h; ty+=tile_h) {
            const uint16_t yend=(ty+tile_h<g.out_h) ? uint16_t(ty+tile_h) : g.out_h;
            for (uint32_t tx=0; tx<g.out_w; tx+=tile_w) {
                const uint16_t xend=(tx+tile_w<g.out_w) ? uint16_t(tx+tile_w) : g.out_w;
                for (uint16_t py=ty; py<yend; ++py) {
                    for (uint16_t px=tx; px<xend; ++px) {
                        int8_t maxval=-128;
                        int32_t sum=0;
                        uint16_t n=0;
                        for (uint8_t wy=0; wy<g.pool_kh; ++wy) {
                            for (uint8_t wx=0; wx<g.pool_kw; ++wx) {
                                const uint32_t y=uint32_t(py)*g.pool_sh+wy;
                                const uint32_t x=uint32_t(px)*g.pool_sw+wx;
                                int32_t acc=bias;
                                for (uint16_t ic=0; ic<in.channels(); ++ic) {
                                    for (uint8_t ky=0; ky<k.kh; ++ky) {
                                        const int32_t iy=int32_t(y)*k.sh+ky-k.pt;
                                        if (iy<0 || iy>=in.height()) continue;
                                        for (uint8_t kx=0; kx<k.kw; ++kx) {
                                            const int32_t ix=int32_t(x)*k.sw+kx-k.pl;
                                            if (ix<0 || ix>=in.width()) continue;
                                            const uint32_t ii=(uint32_t(ic)*in.height()+uint32_t(iy))*in.width()+uint32_t(ix);
                                            const uint32_t wi=(uint32_t(ic)*k.kh+ky)*k.kw+kx;
                                            int8_t v=0;
                                            if (in_data) v=static_cast<int8_t>(in_data[ii]);
                                            else if ((st=in.get(ii,v))!=OK) return st;
                                            if (!safe_accumulate(acc,int32_t(v)*weights[wi])) return OVERFLOW;
                                        }
                                    }
                                }
                                const int8_t val=apply_activation(requantize(acc,p.quant()),act);
                                if (val>maxval) maxval=val;
                                sum+=val;
                                ++n;
                            }
                        }
                        const int8_t value=pool.type==MAX_POOL ? maxval :
                            (pool.type==AVG_POOL ? rounded_average(sum,n) : maxval);
                        const uint32_t dst=(uint32_t(oc)*g.out_h+py)*g.out_w+px;
                        if ((st=out.put(dst,value))!=OK) return st;
                    }
                }
            }
        }
#if defined(ARDUINO_ARCH_ESP8266)
        // Keep the Wi-Fi task and watchdog serviced between output channels.
        // No numerical state is changed by yielding.
        yield();
#endif
    }
    return OK;
}
Status conv2d_direct(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& p,
                     Conv2D k, Activation act, Pool pool) {
    return convolution(in,out,p,k,act,pool,false);
}
Status conv2d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& p,
              Conv2D k, Activation act, Pool pool) {
#if NOODLE2_DEFAULT_CONV_TILE > 0
    return conv2d_tiled(in,out,p,k,act,pool,
                        NOODLE2_DEFAULT_CONV_TILE,NOODLE2_DEFAULT_CONV_TILE);
#else
    return conv2d_direct(in,out,p,k,act,pool);
#endif
}
Status dwconv2d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& p,
                Conv2D k, Activation act, Pool pool) {
    return convolution(in,out,p,k,act,pool,true);
}
Status pwconv2d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& p,
                uint8_t sh, uint8_t sw, Activation act, Pool pool) {
    return convolution(in,out,p,Conv2D(1,1,sh,sw),act,pool,false);
}
Status conv1d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& p,
              Conv1D k, Activation act, Pool pool) {
    if (in.height()!=1) return BAD_SHAPE;
    if (pool.type!=NO_POOL && (pool.kh != 1 || pool.sh != 1)) return BAD_SHAPE;
    Conv2D spatial(1,k.k,1,k.stride,0,0,k.pad_left,k.pad_right);
    spatial.padding = k.padding;
    return convolution(in,out,p,spatial,act,pool,false);
}

} // namespace noodle2
