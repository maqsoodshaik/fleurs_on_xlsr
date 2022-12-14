import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
import torch

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53"
)
model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav, sample = torchaudio.load("/Users/mohammedmaqsoodshaik/Downloads/sa1.wav")
print(wav.shape)
print(sample)
input_values = feature_extractor(
    wav.squeeze(), return_tensors="pt", sampling_rate=16000
).input_values
with torch.no_grad():
    codevector_probs, codevectors, codevector_idx = model(input_values)
print(codevector_idx.view(-1, 2)[:, 0])
print(codevector_idx.view(-1, 2)[:, 1])
# uk_assent(welcome)
# male(jack)
# tensor([257, 257, 257, 257, 257, 257, 257, 279,  85,  85, 231, 280, 136, 136,
#         136, 136, 280, 280, 280, 280, 280,  51, 129, 209,   6, 257,  31, 279,
#         257,  46,  46, 134,  79, 148,  15, 148,  15,  15, 273, 273, 273, 273,
#         273, 134,  20, 209,  28, 257, 257, 257])
# tensor([172, 172, 172, 172, 172, 172, 226, 226, 226,  49, 251, 153,  33,  33,
#          33,  33, 229, 297, 229, 188, 202, 236, 186, 186, 110,  78, 226, 124,
#         124, 209,  64,  64, 103, 174, 138, 120, 120, 120, 120, 216, 216, 302,
#         302, 302, 302, 110, 172, 172, 172, 172])
# female(maria)
# tensor([257, 257, 257, 257, 257, 257, 279, 279,  85,  85, 188, 119, 119, 119,
#         119, 119, 119, 119, 119, 119, 119, 188, 146,  70,   6,   6, 209,  20,
#         134, 134,  46,  46,  79, 123,  95, 118,  95,  95, 245, 245,  53,  17,
#          17,  17,  17,  17,  17,  17,  17,  20,   6,  28, 257, 257, 257])
# tensor([172, 172, 172, 172, 172, 172, 226, 226, 117, 117, 196, 159, 243, 243,
#         243, 168, 168, 168, 213, 213, 255, 196, 196, 117, 133, 110, 226, 124,
#         264,  64,  40,  64, 123,  76, 103, 103, 103,  76,  76, 272, 174, 196,
#         196, 196, 196, 196, 117, 117, 117, 117, 110, 172, 172, 172, 172])
# female(alisha)
# tensor([257, 257, 257, 257, 257,  28, 279, 279, 279, 275, 278, 278, 278, 119,
#         119, 119, 119, 119, 115, 278, 278, 188, 188, 146, 146,  70,  20,  31,
#          31, 279,  46,  46,  46,  96, 123, 136, 136, 136, 136, 136, 231, 231,
#         231,  51, 129,  15,  15,  15,  15,  15,  17,  17,  17, 279,  17,   6,
#          28,  28, 257, 257])
# tensor([172, 172, 172, 172, 172, 172, 226, 226, 117,  49,  49, 159, 159, 243,
#         243, 243, 168, 168, 213, 213,  92,  92, 196, 186, 117, 117, 110, 110,
#         124, 124, 264,  40,  64, 171, 120, 216, 123, 123, 103, 103,  76,  76,
#          76,  76, 174, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 302,
#         110, 172, 172, 172])
# male(steven)
# tensor([257, 257, 257, 257, 257, 257, 257, 279,  85, 128, 231, 280, 280, 231,
#         280,  34,  34,  34,  34,  34,  34,  34,  15,   6,  20,  31,  31,  31,
#         134, 134,  46,  46,  79,  15,  15,  15, 273, 134, 134,  20,  20,  20,
#          20,  20,  20,  20,  20,  20,  20,  20, 209,  28, 257, 257, 257, 257,
#         257])
# tensor([172, 172, 172, 172, 172, 172, 226, 226,  49, 251, 251, 120, 120, 138,
#         260, 120, 229, 309, 309, 202, 202, 296, 186, 186, 110, 289, 226, 264,
#         264, 243, 292,  64,  76, 302, 302, 302, 302, 302, 117, 117, 117, 117,
#         117, 117, 117, 117, 117, 117, 117, 110, 110, 110, 172, 172, 172, 172,
#         172])
# us(female(alisha))
# tensor([257, 257, 257, 257,  28,  28, 279,  85,  17,  17,  85,  95,  95,  95,
#          95,  95, 229, 229, 229, 229, 229, 229,  51,   6,  31,  31,  31,  31,
#         279, 257,  28,  79,  34,  34,  34,  34,  34,  15,  15,  15,  15,  15,
#          15,  15,  15, 273, 134,  20,   6,  28,  28, 257, 257])
# tensor([172, 172, 172, 226,  46, 226, 226, 117, 117, 117, 196, 196,  92,  92,
#         213, 213, 213, 213,  92,  92,  92, 196, 117, 117, 110, 110, 226, 226,
#         264, 243, 292, 174,  76,  76,  79, 243, 243,  79, 196, 196, 117, 117,
#         117, 117, 117, 117, 117, 117, 110, 289, 172, 172, 172])


# timit(sa1.wav)
# tensor([257, 257, 257, 257, 257, 257,  31, 160,  31,  31,  31,  31,  46,  46,
#          46,  46,  46, 134, 134, 134, 134, 257,  28,  28, 160, 209, 209,  31,
#         160,  31,  66,  33,  33,  33,  33,  95,  95, 136, 136, 245,  12,  12,
#          96, 123, 123, 118, 118, 118, 118, 118,  12, 209,  33, 118,  95,  95,
#          95, 245,  12, 209,  20,  45, 123,  95,  95,  95,  95,  95, 136,  11,
#         273,   6, 279, 257,  33,  82,  82, 113, 113, 113, 113, 113,  95,  95,
#          95,  95, 245, 136,  51, 257,  45, 103, 231, 231,  34,  34,  34,  34,
#         129,  70,   6,  31, 209,  85, 231, 136,  95,  95,  95,  12,  82, 113,
#         113, 113, 113,  82, 103, 123, 231, 128,  11, 257, 160, 279, 128, 231,
#         231, 136, 136, 136,  95, 245,  12,  96,  33,  33,  33,  33,  33,  66,
#          29,  46, 279, 275, 212, 212, 212, 212,  34,  34, 129, 277, 279,  79,
#         119, 119, 119, 119,   4,  11, 273, 273, 279, 279,  34,  34,  34,  34,
#          34,  34,  51, 128, 129, 129,  34,  34,  51,  51, 136, 280, 280,  34,
#         128, 275,  11, 273, 273, 273, 273, 134, 134,  28, 160, 209,   6,  20,
#           6, 209])
# tensor([ 46,  46,  46,  46, 289,  46, 226,  46,  18, 199, 199, 199, 124, 140,
#         140, 140, 209, 209, 209, 209, 209, 209, 209, 209, 209, 140,  83,  83,
#         289, 310, 310, 140, 140,  66,  97, 247, 185, 185, 185, 102, 102,  15,
#         297, 297,  98,  98, 270, 270,   1,   1, 260, 302, 214, 222,  15, 309,
#         260, 123, 120, 236, 236, 214, 200, 270,  90, 128,  90,   1, 292,  64,
#         123, 124, 226,  46,  18, 124,   3,   3,   3,  66,  83, 228, 228, 119,
#         301, 306, 162, 162, 174, 251, 214, 251, 188, 253, 253, 196, 196, 196,
#          49, 133, 306, 200, 215, 101, 138, 188, 222, 185, 166, 183, 230,  66,
#          66,   3,  83,  83, 307, 247, 102, 296, 306,  92, 196, 196,  92, 213,
#         168, 168, 168, 128, 128,  90,   1, 124, 264,   3,  66,  66, 140,  46,
#          46, 238,  92, 213, 213, 168, 168, 168, 128, 128,  90, 292, 310, 171,
#          64, 103, 103, 103, 103, 103, 292, 128, 168, 168, 213, 213, 213, 213,
#         213, 213, 101, 138, 188, 102, 166, 166, 166, 166, 166, 166, 166, 166,
#         102,  15,  15, 309, 123, 123, 123, 138,  64,  64,  40,  64,  40,  83,
#          66,  66])
# 0 9640 h#
# 9640 11240 sh
# 11240 12783 iy
# 12783 14078 hv
# 14078 16157 ae
# 16157 16880 dcl
# 16880 17103 d
# 17103 17587 y
# 17587 18760 er
# 18760 19720 dcl
# 19720 19962 d
# 19962 21514 aa
# 21514 22680 r
# 22680 23800 kcl
# 23800 24104 k
# 24104 26280 s
# 26280 28591 uw
# 28591 29179 dx
# 29179 30337 ih
# 30337 31880 ng
# 31880 32500 gcl
# 32500 33170 g
# 33170 33829 r
# 33829 35150 iy
# 35150 37370 s
# 37370 38568 iy
# 38568 40546 w
# 40546 42357 aa
# 42357 45119 sh
# 45119 45624 epi
# 45624 46855 w
# 46855 48680 aa
# 48680 49240 dx
# 49240 51033 er
# 51033 52378 q
# 52378 54500 ao
# 54500 55461 l
# 55461 57395 y
# 57395 59179 iy
# 59179 60600 axr
# 60600 63440 h#
