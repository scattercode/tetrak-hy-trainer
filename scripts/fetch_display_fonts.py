#!/usr/bin/env python3
"""Fetch brief 019's display faces: the letter shapes of modern print.

Every face the renderer has drawn in since v0 is a book or official text
face (GHEA Grapalat and Mariam, Arian AMU, Noto, Mshtakan). Modern print --
flyers, magazines, programmes, covers -- sets headings and quotations in
display faces where letters are drawn in ways an Armenian reader takes in
their stride: a forward-hooked ``ն`` (the 2021 epigraph v6 read as ``ճ``
throughout), an angular ``ճ``, ``ու`` joined, capitals at lowercase code
points. This fetches the faces brief 019 chose to cover those shapes into
``runs/fonts-display/``, beside rather than among the book faces, so the
pre-train's face set does not change underneath brief 014.

Two folders:

- ``train/``: faces ``render_synthetic_set.py --font-dir`` draws in.
- ``heldout/``: four whole families no training set may draw in
  (Kelvinch, Iosevka Etoile, Sharqayin, Catrinity), rendered by
  ``render_face_testset.py`` so that reading a face the model has never
  seen is a number rather than a hope.

``faces.tsv`` records each file's family, folder, source and recorded
licence; the renderer reads the family column to sample by family rather
than by file.

Every file is pinned by SHA-256 (``tetrak_hy_trainer.fontsource``). Variable
fonts are pinned as sources and cut into fixed instances here, because
Pillow draws only a variable font's default instance: the Noto variable
files would otherwise add nothing the static Regulars did not.

Licences, verified 30 October 2026 from each file's name table (13/14) and
the source's licence file: OFL for most; GPLv3 with the font exception for
GNU FreeFont; Bitstream Vera and public domain for DejaVu; Apache-2.0 in
Mardoto's repository beside OFL in its files. GAA Gayane, Galiver Sans, the
Braind and Renderforest faces, Calama Condensed and Arti are used on
fonter.am's stated terms, accepted for this project on 30 October 2026;
``TERMS-fonter.md`` is written beside them with the date. We never
redistribute a font, only images rendered with it; credit the designers in
any dataset or model card that includes those images.

The fonter.am faces come from the fonts-armenian-extra collection on
GitHub, pinned to a commit, since fonter.am's own links are signed and
expire. Do not script around that signing.

Prerequisites: this repository's venv (requests and fontTools).

Run from the repo root:

    python scripts/fetch_display_fonts.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from urllib.parse import quote

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tetrak_hy_trainer import charset, fontsource, synth  # noqa: E402
from tetrak_hy_trainer.fontsource import Face  # noqa: E402

DEST = REPO / "runs" / "fonts-display"

HELD_OUT = frozenset({"Kelvinch", "Iosevka Etoile", "Sharqayin", "Catrinity"})

_GF = "https://raw.githubusercontent.com/google/fonts/6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec/ofl/"
_MARDOTO = (
    "https://raw.githubusercontent.com/vahanhovh/mardoto/"
    "4bcb915be4b616f2097386058627709fc66fceac/fonts/ttf/"
)
_SHARQAYIN = (
    "https://raw.githubusercontent.com/erankyun/Sharqayin/"
    "b3ce4b23665ba0d4d7f8911c50fa74f6733e3dfd/OpenType-TT/"
)
_HAR8 = (
    "https://raw.githubusercontent.com/HarutyunSoghomonyan/Har8-Black-Ops-One/"
    "9927955b56f6092ded2cda3fa81278a2da6213aa/Fonts/"
)
_WEBSTART = (
    "https://raw.githubusercontent.com/Webstart-LLC/webstart-armenian-font/"
    "e576266b5669568679187e3136a2239d72408e16/"
)
_FREEFONT = "https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip"
_DEJAVU = (
    "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
    "version_2_37/dejavu-fonts-ttf-2.37.zip"
)
_IOSEVKA = (
    "https://github.com/be5invis/Iosevka/releases/download/v34.9.1/PkgTTF-Iosevka{}-34.9.1.zip"
)
_MIRROR = (
    "https://codeload.github.com/lecramyajiv/fonts-armenian-extra/tar.gz/"
    "387f6ec024db5a8bf9a819deda146a8871dd64dc"
)
_MIRROR_ROOT = "fonts-armenian-extra-387f6ec024db5a8bf9a819deda146a8871dd64dc"


# SHA-256 of every pinned file, by the name it is saved as. Re-pin with
# --pin only after looking at what changed upstream.
PINS = {
    "Mardoto-Light.ttf": "08cae7eae992428de650795f6bfffa3397cdf64092bb2a064d6c4f976c1ac9f8",
    "Mardoto-LightItalic.ttf": "031600b98bc71cb14e1088c6960827d4f84948604214507f09402ef8aecd61a5",
    "Mardoto-Regular.ttf": "2e74ec5d9f9518f69b25222fc89e8ac2410bfaf99242efb9f11f6339ff33ee69",
    "Mardoto-Italic.ttf": "52e1d8ced9ba1c4982808a165c4fc21656846aac2f66f36ed52e694478e2a3d6",
    "Mardoto-Medium.ttf": "6825fc577124be5c82a6100af45affe122e63d8e55494b42617e931649eec0e4",
    "Mardoto-MediumItalic.ttf": "98a67122a93a8a3e423ed67de054e13354997608bbec1fe42ae0fc5815b33224",
    "Mardoto-Bold.ttf": "ed8de3952cffdd3f84ffa750438012b6bf7cdc170eb1cfb99d77fe5fe2d61e21",
    "Mardoto-BoldItalic.ttf": "34cec3967574d2ac76c74537d8a2566e7ce3782b423b0e858261950b51040cdc",
    "Mardoto-Black.ttf": "070fc3d4e522c0bb541a517f18c4655080d89cb3979d6587b24ab1dee3b1112c",
    "Mardoto-BlackItalic.ttf": "d0676131887f7727551d3048e7e727ee1e44212c8f75de1d7561f809df2856a6",
    "Sharqayin-Regular.ttf": "98862dc307f3b90ecce5247b9142c586e953770792930a162561a47db61bf1c9",
    "Sharqayin-Bold.ttf": "c97e55e8d290a86f4720bf3589becfe8fd4ad360995f3ed549ac3ac7e4b348b6",
    "Sharqayin-Black.ttf": "f72520680c7f6fac5d4bbdf34c10713d4237ee0fa35908b5f7899da4def707ab",
    "Har8-BlackOpsOne-Regular.ttf": "b82f52aaa3f1fc09b161998df91877b33d291c5f874041ababf26fddd030b63b",
    "Webstart-Light.ttf": "6454db9e71c345654a770d36cf641f7e4f535a467c13a9e0db5f8e9ef71d8180",
    "Webstart-Regular.ttf": "e2070f1725a00284e124ec5d5ab9462095871da3810e7a44ef0737776a774ae1",
    "Webstart-Bold.ttf": "4156ba3ac67ba4e9be5afebb2be37ae942ff0fbe97a0968f5a108260ca8af5d1",
    "Webstart-Black.ttf": "dac6d4beea3b847e622200a80e59247a4cf47dfb84601156d420a799e8c757a9",
    "FreeSans.ttf": "c80858440d8fb618e0ac5ff6f16251dbfa6b3316f00f3cdd17d477297dd87b04",
    "FreeSansOblique.ttf": "43ef481308dfb3aa1103a75699cb4976139b6339eda0901df69a25533f07cf6b",
    "FreeSansBold.ttf": "982534a3731416a15e2756601721f26053f68bf4239011550f3dd23ce6308215",
    "FreeSansBoldOblique.ttf": "deb48d677df0b521793df8da28729316bfcc255309aeb6dacfbd38241ac9b7ba",
    "FreeSerif.ttf": "658563c732eafdc851de2b3f38b27d0afcde5674b2f93440971b9693438c6494",
    "FreeSerifItalic.ttf": "1cef05c1ea5d88f0e06b212bdb0510edf65c5456523dd81ddccfe9847a77284f",
    "FreeSerifBold.ttf": "973791e2bb5ef4ba7ac9e7423f8a8d8441468da3a37c7e38ab38256011f5f261",
    "FreeSerifBoldItalic.ttf": "38e39596c9daec677633761324216ffda969c65b9515cf273b001521f22cb2b3",
    "DejaVuSans.ttf": "7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954",
    "DejaVuSans-Bold.ttf": "e6476c1b80502924294eed40894c5b18e06c181444ca953e5334262df9c27724",
    "DejaVuSans-Oblique.ttf": "4af75fa16ee6d3ad43e1ecec41862c24954af26a55c6bb1ebb27bd486a50f5f4",
    "DejaVuSans-BoldOblique.ttf": "eb436dca0c2594b73d8b603b892e374fdfd8d885d25ffb4f18df4c4c0b49e50f",
    "DejaVuSansCondensed.ttf": "8550cd5ca1acb65a8fc7877c46939cd0b4d909f8e7bc1e24716873d918c5e549",
    "DejaVuSansCondensed-Bold.ttf": "38098d0b8edd8430a0f53fb2831b5d36037563a998a620d3c87a4d7591766d20",
    "DejaVuSerif.ttf": "42d1edeb7952f31b1f96d767ed7030b08a39e0c372b0071641518864e2bffb51",
    "DejaVuSerif-Bold.ttf": "c47b5527bcdc8dcf9ea8c77054454c5a884beaca2f44851a2a823ee639cbf07f",
    "DejaVuSerif-Italic.ttf": "2e39b1d50f90b933b00c7bb54a96afd3f86419b3d717c7cf202e36f2d4973e47",
    "DejaVuSerif-BoldItalic.ttf": "8d3dd3d31350309042ed226af82b34539bd773518e6107cb352712853ba80308",
    "DejaVuSerifCondensed.ttf": "685dcdfb8daf32a453d3426a4eb8d881c72146ddd999229354fb757511357995",
    "DejaVuSerifCondensed-BoldItalic.ttf": "4cfc5a3700c72413f51c4bb54f09617401a0a63eccb9ff577713e20afa047fa5",
    "IosevkaAile-Light.ttf": "34293117cb9c03411a0f3fb641e11b952ba2f84e17c69b93621ae8df8fece8f8",
    "IosevkaAile-Regular.ttf": "be7e1730d23b0ec9d68966402222ec810c8d6d2a708c1ec3b10e5bab74c5186b",
    "IosevkaAile-Italic.ttf": "fd499d751012a1c18552d57c6d0e839683e86c336f89c5d514a104cd2e6ce843",
    "IosevkaAile-Bold.ttf": "f570e3c62244076941aaa16cbfb57edee4c6d4872b82d0b1b13371020bee9b79",
    "IosevkaAile-BoldItalic.ttf": "366c78ac11fd64c53497a8a3bd25883e1bde6b6b8a7087ce304af6bc1c37b674",
    "IosevkaAile-Heavy.ttf": "dfe5fa807faad361fee370d89b27bacc1bf3127ff2266313fc8b41acb9391a94",
    "IosevkaEtoile-Regular.ttf": "a6bf524be6dc336a1c4034278ac7a405a53adfde99d7ab27b792945e86a62cb3",
    "IosevkaEtoile-Italic.ttf": "a8f55af00be5f782437750af392b50a5fdada912166b06b66d17960f2c850281",
    "IosevkaEtoile-Bold.ttf": "60521e204f4957268ab005f0726f404ee6bda6c77f91d47ae148fefe0f07c07f",
    "IosevkaEtoile-BoldItalic.ttf": "68526f68f2ceebd8ee5c7c9e40c77d67410aeccb8145c585ec2e0e58133c92d1",
    "Kelvinch-Roman.otf": "2b98a238ecd565c508a2ba68ba2bc62a0cb4a7aa50095f9559dd40fce347866a",
    "Kelvinch-Italic.otf": "8bdeae62e7f59e37bf02f8eb44260b9828c0be01cf0eb3eaa97ddd00e47d598f",
    "Kelvinch-Bold.otf": "aeab74b1ef7c9331304b032d3704d22db59e9aa46b7d5552f82037af699ace10",
    "Kelvinch-BoldItalic.otf": "b0ea81c6d2ba1e92d4211cc4bdc175217caa87a5e2d9085cfe4f1899ff5e195f",
    "Catrinity.otf": "0089b007c28692ae34315e06e77143543cae9df758f070752efe4dbd75031d92",
    "Bainsley.ttf": "47f409b94793e8fb82568a16933cecbe4efdb58f2766b48087a8aa827a275adf",
    "Bainsley-Italic.ttf": "bc26925a40ed801e1c6501985462942ecaafac712c9543644cde54c47d0c58d3",
    "Bainsley-Bold.ttf": "57f941422d76ea56f66565b013f4e6e9aed4583bce005a5b3f60471a7522f34f",
    "Bainsley-BoldItalic.ttf": "4713c6e34e7c54cbbae26c6e84a0093a8d63992e3ed40bce44560a9e137811d7",
    "arnamu_serif.ttf": "2fe18e7a1975480f88469855ac447349069743278aa6350202d3a8750db16497",
    "arnamu_serif_italic.ttf": "ecd9e965baedffcda6727a248d1d77d8dd539d17e14c40c70a82bfaeb3119fe2",
    "arnamu_serif_bold.ttf": "89470a872b9280c4361273a570e7a64b6ce909f8d16fc421ec719b66b9ef69e0",
    "arnamu_serif_italic_bold.ttf": "3b983eec31e8e291365bb758f336155f585cd8705099506a16b049cfc4d828ae",
    "Arian_Grqi_U.ttf": "6d96f4a83716a65d438a8440baab0bd8929689c369725d16961fc546e30a30e5",
    "Arian_Grqi_i_U.ttf": "24d2eccfc3cefdead81853030c8c0159828f08c104de41529b808594224518d9",
    "Kurland.otf": "630aa426b4b346db452e0f72a4eb084165b84748b8225c31dd58ca883317f505",
    "Quivira.otf": "03c2f593b224efffceac6f7f3090cdc935f0817399fcc607e4c7cc1b097b3b5a",
    "Oldtimer.ttf": "bf144623db5f88e83c3455fc5bc800fcd49114af7c382a351beaa269192fecff",
    "OpenGOSTtypeA-Regular.ttf": "f77d96f832b7c902d42fc391fddf15b622c1eadeb263dd31b926888d72e0773a",
    "OpenGOSTtypeB-Regular.ttf": "8307a636e921a3be10d36034581caafd8cf36cb85909b8add21d22f71fcde1f8",
    "Montserratarm-Light.ttf": "15a11edc9d23a8b25c1faecb6891683706c25f24cdccfd67ab1759d68b3ddec0",
    "Montserratarm-Regular.ttf": "b2403c8716de62030a0fb3a37e362a3b1b29d1c4fb6986955a0b890b40275312",
    "Montserratarm-Medium.ttf": "5bcbf707bc4f91365b43f3147d0de9dfeaa8336edff355ad8b76a02a2d1c5f6d",
    "Montserratarm-Bold.ttf": "474321d18bfea1c351da525d7ca2ad4c89151c53f3d9161069bc4c1938ad95b0",
    "Montserratarm-ExtraBold.ttf": "63a63f974dccf9453789cfa5f6e3f40e034b35f1bb0b8a1f676ef77b6b29392d",
    "Montserratarm-Black.ttf": "73a144ca2a5952925042498fef3406b8dec9a8e28b8a7ccfae3047d325b4d369",
    "GAA_Gayane_Reg.otf": "31ea4d1ed32ec4c7646c33cacb2c8366ff6e2b75a65126013fed57f7d671dc7e",
    "GAA_Gayane_Reg_it.otf": "384349f5278f9aa4b5718b187b38bee11123218a81bfd4b57cd21e6a8ea4668e",
    "GAAGayane-Medium.otf": "56928c2711349b929798c0cad7246019f5edfd6f7f1700e454f44a62cc5ff3ed",
    "GAAGayane-Medium_it.otf": "ed50efeb8bf5d9641294b39064912777779942bd85269c4a4397437043f42ed1",
    "GAAGayane-SemiBold.otf": "7d72c8f84ce4f660c855043a9e8180b67b56ce4f5157de6cf97eccff3d5ddd18",
    "GAAGayane-SemiBold_It.otf": "0a211923728f3e279a68753de20ab77420ce5bcad305150b70550ef5ae20d72c",
    "GAAGayane-DemiBold.otf": "593ec8f2812ffb61638043e353b60196b8b83ca530bc343695b4b9af0f77618f",
    "GAAGayane-DemiBold_it.otf": "e3198d1e4a1a45ce6f9ca110ec8d9ed02a62d98c85e9f8f19caa03e2fff96caa",
    "GAAGayane-Bold.otf": "c17ff2bbfc22f11330098a831c7ea1c9f890774290a61e1742e49261dc70a4d3",
    "GAAGayane-Bold_it.otf": "90c6a3f1ba1613514e85ceaaf7b657ad96bcf1434a75d6b108e7e1e3984c5e7c",
    "GAA_Gayane_ExtBld.otf": "4cae6adb65c88015f3dbe6f877884a6df55b3734d0b5c02098929ab01ac595ac",
    "GAA_Gayane_ExtBld_it.otf": "2b35aaf56a51f2dc33ef6ed8ded8419e032057c13ee6a61d7009f51fa37ebff5",
    "GaliverSans-lavD.ttf": "496694d5fc734387cbb132ad06e6231f9f67858132a1d81b801fe23394e2acbe",
    "GaliverSansItalic-RDWV.ttf": "45e1e28306a441d764882c9e6ee354a57e5dfd22536fa5691f0d405357dc61dd",
    "GaliverSansBold-nO7V.ttf": "00aa2e2db1b68fbcbefffcbda67ff9d200c5cfe585e4d35abee623ad5a4dda38",
    "GaliverSansBoldItalic-1px0.ttf": "2030e3dca7c3338c430e6a25de8aa1f29b88d1ea0708248cea82e7ea5a9941f0",
    "SGALS.ttf": "23bcc0aa93e844447027b1601ba2dced49e948527f14545fa97c8dcd491e2e10",
    "SGALSI.ttf": "2b3fb3347872d8188fadd9a2202cb16b4863c46e2688d4250b29fb9b9f1b1352",
    "SGALSB.ttf": "9606b65e7636add901c6ec1b483d678e130a90c550a789af764a89cbc3bea48f",
    "SGALSBI.ttf": "6497e558ac6b19f3c49afcd7d60160a600ed3f749e902a1799cf84abb650c1d6",
    "Bokonique-Light.otf": "6afb1b0d2f24f160c8650a0ea694b476b4a8f51ffdbad4ccb9fb687394c99e8e",
    "Bokonique-Light-Italic.otf": "bb4e16a8269fd3d64f1d441b4a7a11eec0b0a8e97552f71e8583f8aae0c3601c",
    "Bokonique-Regular.otf": "9f7d4f3fd4d20ad32c1d6bf2b5eee0a76856c1fd94324ee444ada6e87a076c73",
    "Bokonique-Regular-Italic.otf": "bc39060ca28ad9523b28b41047c40cedfb95c7536aa0472649dc896ecd7c77dd",
    "Bokonique-Bold.otf": "86cdc1014ca892536771c8ef1ec265e74e28ab6924852a8f5d0bda559511b7a2",
    "Bokonique-Bold-Italic.otf": "8d3479676d118b3dbefe6d41e3b1a1f75e3689b220098e28aecf94a7725bd509",
    "Nicolo-Regular.otf": "83ab550ee32ed6edf30d3ff75a59d3683d107484999089013f6db024e0d224ed",
    "Nicolo-Italic.otf": "cb4132c9ada2a6f0f1e7a33f0f55b2b581786e3f1557df2e611d9ea6d5615614",
    "NicoloBold.otf": "e49fdad279d438cf64b0dee43b7c4f1d40047a80ef4277fcfe13f3b1ac4745a8",
    "Nicolo-BoldItalic.otf": "f48597c72651f3ad7531ff1a9d93e91e6ed1174f7b4110145fbd895191afc15a",
    "Nortar-Regular.otf": "ace89a6ccf09c22003ad2e73266e480c5a40c5f2c4c08d1e5dc63123f0291d7e",
    "Nortar-Regular-Italic.otf": "90e3fe3e5863daab1486e24b369166e580e7378fa072a9ee7bfc3016f1aabd47",
    "Nortar-Bold.otf": "11b8c5d3a95f85d15f78913bce7fecf6a28fe798090d21ff72e53c8525974c55",
    "Nortar-Bold-Italic.otf": "35ba16780eaf246a6592db92fce7b46371344ae8b5d530fa31f9dbd2b6e56487",
    "Yerevanyan-Regular.otf": "2f4313851dfc180a362aae2ca5c583afc4d218b8a2e7be5f31f93eb93993b69c",
    "Yerevanyan-Regular-Italic.otf": "eb491c523ffbebe0877d06641d0da6880a5e326866f4e191e2064faaa92a65d4",
    "Yerevanyan-Bold.otf": "5c07648fd14cbfedc309410f2ddaddada36c8c3e08dd24e3c8c0683ac3b2b7c2",
    "Yerevanyan-Bold-Italic.otf": "940588a155fd86ba8d77145c8f5db4118d0a4979fc9919b70a6dac5427b2ac35",
    "Grqalir-Regular.otf": "9ad0930b651dd108be74d82a70522a3219edb8791362e951985e9fe21a21ab44",
    "Grqalir-Italic.otf": "89e26173bbe7f354534e3fe4f59832285144c4653797c700647d09b091f9c96b",
    "Grqalir-Bold.otf": "ab7b5fc3b7873273217581d62e61bdfc4200ebf0f6671c22a6a716e13b5bf6b1",
    "Grqalir-Bold-Italic.otf": "c01c121dd36b3b2ad7ce870b169ffedde60abacc508f9048a1516380024faa59",
    "BraindAmanor.otf": "da92de29ea2f82a713c4f33562e29728644ebccc3ee908d371e547b0a7a5bf2a",
    "Braind-Zbans.otf": "78c139dacacb754fed048919862c7d8e081b424f3070a5356d0a427a72ba2e40",
    "BraindYerevan.otf": "45f345fc982022893a943b303eaa63bbd376ea7cc5af28eaa7769008365d1dad",
    "BraindGyumri.otf": "0f90fb9d6c413c31bda7b0474b3c7cc4941b165b780b6043196669a1ab377b1b",
    "Braind-Republic-Regular.otf": "935c8a2209e7e65a1176b39564eee1547729b7ee8686b36044bb7f738606b141",
    "Calama.ttf": "7f909d135a942ba96b1a18bc9b4caddd35792e02aefb7000f49a75d644d8739a",
    "arti-v2.otf": "b78bd237fe84d1b6c06ed5abc1824e5fced6145129d7158f48640a1381e3e1b5",
    "NotoSansArmenian[wdth,wght].ttf": "0870908d8318435a5daf1cd280ae15063f990cd9ad60f3e94c734ce9e1ffef71",
    "NotoSerifArmenian[wdth,wght].ttf": "9354b74384d093d59f9a4ad958e188808d58b6f15debeaab18e07bc432f8ce0c",
    "Handjet[ELGR,ELSH,wght].ttf": "9262749e8bb0b73ebcae0e20428689c3c59576eebeb6c4e1020300d2d41bdf4d",
    "GoogleSans[GRAD,opsz,wght].ttf": "d0a87d835a944b8b40d0e82a5651bb59ab97b936a2aeed5946eb57e7b2a3a90a",
    "GoogleSans-Italic[GRAD,opsz,wght].ttf": "b6662abd2131cad1b248750149a7f2b644f871ebcc9af7fe4cd1135cd290b6d1",
}


def _raw(family: str, base: str, filename: str) -> Face:
    return Face(family, filename, base + quote(filename), None, PINS.get(filename, ""))


def _zip(family: str, url: str, member: str) -> Face:
    return Face(family, Path(member).name, url, member, PINS.get(Path(member).name, ""))


def _mirror(family: str, member: str) -> Face:
    name = Path(member).name.replace(" ", "-")
    return Face(family, name, _MIRROR, f"{_MIRROR_ROOT}/{member}", PINS.get(name, ""))


# Static faces, as published. Thin and ExtraLight weights are left out: at
# the renderer's smallest sizes, degraded, their strokes vanish, and the
# label would then name letters the crop no longer shows.
FACES = (
    _raw("Mardoto", _MARDOTO, "Mardoto-Light.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-LightItalic.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-Regular.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-Italic.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-Medium.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-MediumItalic.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-Bold.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-BoldItalic.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-Black.ttf"),
    _raw("Mardoto", _MARDOTO, "Mardoto-BlackItalic.ttf"),
    _raw("Sharqayin", _SHARQAYIN, "Sharqayin-Regular.ttf"),
    _raw("Sharqayin", _SHARQAYIN, "Sharqayin-Bold.ttf"),
    _raw("Sharqayin", _SHARQAYIN, "Sharqayin-Black.ttf"),
    _raw("Har8 Black Ops One", _HAR8, "Har8-BlackOpsOne-Regular.ttf"),
    _raw("Webstart", _WEBSTART, "Webstart-Light.ttf"),
    _raw("Webstart", _WEBSTART, "Webstart-Regular.ttf"),
    _raw("Webstart", _WEBSTART, "Webstart-Bold.ttf"),
    _raw("Webstart", _WEBSTART, "Webstart-Black.ttf"),
    *(
        _zip("FreeSans" if "Sans" in name else "FreeSerif", _FREEFONT, f"freefont-20120503/{name}")
        for name in (
            "FreeSans.ttf",
            "FreeSansOblique.ttf",
            "FreeSansBold.ttf",
            "FreeSansBoldOblique.ttf",
            "FreeSerif.ttf",
            "FreeSerifItalic.ttf",
            "FreeSerifBold.ttf",
            "FreeSerifBoldItalic.ttf",
        )
    ),
    *(
        _zip(
            ("DejaVu Serif" if "Serif" in name else "DejaVu Sans")
            + (" Condensed" if "Condensed" in name else ""),
            _DEJAVU,
            f"dejavu-fonts-ttf-2.37/ttf/{name}",
        )
        for name in (
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique.ttf",
            "DejaVuSansCondensed.ttf",
            "DejaVuSansCondensed-Bold.ttf",
            "DejaVuSerif.ttf",
            "DejaVuSerif-Bold.ttf",
            "DejaVuSerif-Italic.ttf",
            "DejaVuSerif-BoldItalic.ttf",
            "DejaVuSerifCondensed.ttf",
            "DejaVuSerifCondensed-BoldItalic.ttf",
        )
    ),
    *(
        _zip("Iosevka Aile", _IOSEVKA.format("Aile"), f"IosevkaAile-{style}.ttf")
        for style in ("Light", "Regular", "Italic", "Bold", "BoldItalic", "Heavy")
    ),
    *(
        _zip("Iosevka Etoile", _IOSEVKA.format("Etoile"), f"IosevkaEtoile-{style}.ttf")
        for style in ("Regular", "Italic", "Bold", "BoldItalic")
    ),
    *(
        _mirror(family, member)
        for family, member in (
            ("Kelvinch", "OTF/Kelvinch-Roman.otf"),
            ("Kelvinch", "OTF/Kelvinch-Italic.otf"),
            ("Kelvinch", "OTF/Kelvinch-Bold.otf"),
            ("Kelvinch", "OTF/Kelvinch-BoldItalic.otf"),
            ("Catrinity", "OTF/Catrinity.otf"),
            ("Bainsley", "TTF/Bainsley.ttf"),
            ("Bainsley", "TTF/Bainsley-Italic.ttf"),
            ("Bainsley", "TTF/Bainsley-Bold.ttf"),
            ("Bainsley", "TTF/Bainsley-BoldItalic.ttf"),
            ("Arian AMU Serif", "TTF/arnamu_serif.ttf"),
            ("Arian AMU Serif", "TTF/arnamu_serif_italic.ttf"),
            ("Arian AMU Serif", "TTF/arnamu_serif_bold.ttf"),
            ("Arian AMU Serif", "TTF/arnamu_serif_italic_bold.ttf"),
            ("Arian Grqi", "TTF/Arian_Grqi_U.ttf"),
            ("Arian Grqi", "TTF/Arian_Grqi_i_U.ttf"),
            ("Kurland", "OTF/Kurland.otf"),
            ("Quivira", "OTF/Quivira.otf"),
            ("Oldtimer", "TTF/Oldtimer.ttf"),
            ("OpenGOST", "TTF/OpenGOSTtypeA-Regular.ttf"),
            ("OpenGOST", "TTF/OpenGOSTtypeB-Regular.ttf"),
            ("Montserrat arm", "TTF/Montserratarm-Light.ttf"),
            ("Montserrat arm", "TTF/Montserratarm-Regular.ttf"),
            ("Montserrat arm", "TTF/Montserratarm-Medium.ttf"),
            ("Montserrat arm", "TTF/Montserratarm-Bold.ttf"),
            ("Montserrat arm", "TTF/Montserratarm-ExtraBold.ttf"),
            ("Montserrat arm", "TTF/Montserratarm-Black.ttf"),
            # fonter.am's stated terms (see the module docstring).
            ("GAA Gayane", "OTF/GAA_Gayane_Reg.otf"),
            ("GAA Gayane", "OTF/GAA_Gayane_Reg_it.otf"),
            ("GAA Gayane", "OTF/GAAGayane-Medium.otf"),
            ("GAA Gayane", "OTF/GAAGayane-Medium_it.otf"),
            ("GAA Gayane", "OTF/GAAGayane-SemiBold.otf"),
            ("GAA Gayane", "OTF/GAAGayane-SemiBold_It.otf"),
            ("GAA Gayane", "OTF/GAAGayane-DemiBold.otf"),
            ("GAA Gayane", "OTF/GAAGayane-DemiBold_it.otf"),
            ("GAA Gayane", "OTF/GAAGayane-Bold.otf"),
            ("GAA Gayane", "OTF/GAAGayane-Bold_it.otf"),
            ("GAA Gayane", "OTF/GAA_Gayane_ExtBld.otf"),
            ("GAA Gayane", "OTF/GAA_Gayane_ExtBld_it.otf"),
            ("Galiver Sans", "TTF/GaliverSans-lavD.ttf"),
            ("Galiver Sans", "TTF/GaliverSansItalic-RDWV.ttf"),
            ("Galiver Sans", "TTF/GaliverSansBold-nO7V.ttf"),
            ("Galiver Sans", "TTF/GaliverSansBoldItalic-1px0.ttf"),
            ("Galiver Sans Obliques", "TTF/SGALS.ttf"),
            ("Galiver Sans Obliques", "TTF/SGALSI.ttf"),
            ("Galiver Sans Obliques", "TTF/SGALSB.ttf"),
            ("Galiver Sans Obliques", "TTF/SGALSBI.ttf"),
            ("Bokonique", "OTF/Bokonique-Light.otf"),
            ("Bokonique", "OTF/Bokonique-Light Italic.otf"),
            ("Bokonique", "OTF/Bokonique-Regular.otf"),
            ("Bokonique", "OTF/Bokonique-Regular Italic.otf"),
            ("Bokonique", "OTF/Bokonique-Bold.otf"),
            ("Bokonique", "OTF/Bokonique-Bold Italic.otf"),
            ("Nicolo", "OTF/Nicolo-Regular.otf"),
            ("Nicolo", "OTF/Nicolo-Italic.otf"),
            ("Nicolo", "OTF/NicoloBold.otf"),
            ("Nicolo", "OTF/Nicolo BoldItalic.otf"),
            ("Nortar", "OTF/Nortar-Regular.otf"),
            ("Nortar", "OTF/Nortar-Regular Italic.otf"),
            ("Nortar", "OTF/Nortar-Bold.otf"),
            ("Nortar", "OTF/Nortar-Bold Italic.otf"),
            ("Yerevanyan", "OTF/Yerevanyan Regular.otf"),
            ("Yerevanyan", "OTF/Yerevanyan Regular Italic.otf"),
            ("Yerevanyan", "OTF/Yerevanyan Bold.otf"),
            ("Yerevanyan", "OTF/Yerevanyan Bold Italic.otf"),
            ("Grqalir", "OTF/Grqalir-Regular.otf"),
            ("Grqalir", "OTF/Grqalir-Italic.otf"),
            ("Grqalir", "OTF/Grqalir-Bold.otf"),
            ("Grqalir", "OTF/Grqalir-Bold Italic.otf"),
            ("Braind Amanor", "OTF/BraindAmanor.otf"),
            ("Braind Zbans", "OTF/Braind Zbans.otf"),
            ("Braind Yerevan", "OTF/BraindYerevan.otf"),
            ("Braind Gyumri", "OTF/BraindGyumri.otf"),
            ("Braind Republic", "OTF/Braind Republic Regular.otf"),
            ("Calama Condensed", "TTF/Calama.ttf"),
            ("Arti", "OTF/arti-v2.otf"),
        )
    ),
)

# Variable sources, pinned, and the fixed instances cut from each. Axes left
# out stay at their defaults.
VARIABLE = (
    (
        _raw("Noto Sans Armenian", _GF + "notosansarmenian/", "NotoSansArmenian[wdth,wght].ttf"),
        {
            "Bold": {"wght": 700},
            "Black": {"wght": 900},
            "Condensed": {"wdth": 62.5},
            "CondensedBold": {"wdth": 62.5, "wght": 700},
            "SemiCondensedLight": {"wdth": 75, "wght": 300},
        },
    ),
    (
        _raw("Noto Serif Armenian", _GF + "notoserifarmenian/", "NotoSerifArmenian[wdth,wght].ttf"),
        {
            "Bold": {"wght": 700},
            "Black": {"wght": 900},
            "Condensed": {"wdth": 62.5},
            "CondensedBold": {"wdth": 62.5, "wght": 700},
            "SemiCondensedLight": {"wdth": 75, "wght": 300},
        },
    ),
    (
        _raw("Handjet", _GF + "handjet/", "Handjet[ELGR,ELSH,wght].ttf"),
        # Not ELSH 0: an instance there draws no ink at all, though its cmap
        # is full, so every crop rendered in it would be blank and labelled.
        {
            "Regular": {"wght": 400},
            "LightE8": {"wght": 300, "ELSH": 8},
            "SemiBoldE16Double": {"wght": 600, "ELSH": 16, "ELGR": 2},
        },
    ),
    (
        _raw("Google Sans", _GF + "googlesans/", "GoogleSans[GRAD,opsz,wght].ttf"),
        {"Regular": {"wght": 400}, "Bold": {"wght": 700}},
    ),
    (
        _raw("Google Sans", _GF + "googlesans/", "GoogleSans-Italic[GRAD,opsz,wght].ttf"),
        {"Italic": {"wght": 400}, "BoldItalic": {"wght": 700}},
    ),
)

FONTER_TERMS = """\
# fonter.am terms for the display faces

Accepted for tetrak-hy-trainer by Stevie Masters on 30 October 2026 for:
GAA Gayane (Gayane Andreasyan), Galiver Sans and Galiver Sans Obliques
(Josen Fonts), Bokonique, Nicolo, Nortar, Yerevanyan and Grqalir (Hrach
Hovhannissyan, Renderforest), Braind Amanor, Zbans, Yerevan, Gyumri and
Republic (Edward Kankanyan), Calama Condensed (Ben Karamyan) and Arti
(Khajag Apelian for Hamazkayin).

The terms as fonter.am states them for these fonts, copied on that date:

> Use it in personal and paid work: logos, print, packaging, film, apps.
> Install it on as many of your own machines as you need.
> Embed it in documents, PDFs and exported designs.
> Pass the file on, as long as the author's name stays with it.

fonter.am adds that where an author states different terms, the author's
come first. Credit each designer in any dataset or model card that
includes images rendered in these faces.
"""


def _instances(face: Face, axes_by_style: dict, folder: Path, cache: Path, session) -> list[Path]:
    """Fixed instances of a pinned variable font, cut with fontTools."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    source = fontsource.fetch(face, cache / "variable", cache, session)
    stem = face.family.replace(" ", "")
    out = []
    for style, axes in axes_by_style.items():
        target = folder / f"{stem}-{style}.ttf"
        if not target.exists():
            font = instancer.instantiateVariableFont(TTFont(str(source)), axes, inplace=False)
            folder.mkdir(parents=True, exist_ok=True)
            font.save(str(target))
        out.append(target)
    return out


def main() -> int:
    if "--pin" in sys.argv:
        return pin()
    session = requests.Session()
    session.headers["User-Agent"] = (
        "tetrak-hy-trainer (https://github.com/scattercode/tetrak-hy-trainer)"
    )
    cache = DEST / ".cache"
    rows = []
    for face in FACES:
        folder = DEST / ("heldout" if face.family in HELD_OUT else "train")
        path = fontsource.fetch(face, folder, cache, session)
        rows.append((path, face))
    for face, styles in VARIABLE:
        folder = DEST / ("heldout" if face.family in HELD_OUT else "train")
        rows.extend((path, face) for path in _instances(face, styles, folder, cache, session))
    (DEST / "TERMS-fonter.md").write_text(FONTER_TERMS)

    full = charset.character_list()
    with (DEST / "faces.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("file", "family", "folder", "source", "licence"))
        print(f"{'face':<40} {'folder':<8} {'caps':<5} {'licence (from the font)':<36} gaps")
        for path, face in rows:
            licence = fontsource.licence_of(path).splitlines()[0][:80]
            writer.writerow((path.name, face.family, path.parent.name, face.source, licence))
            gaps = synth.missing_glyphs(str(path), full)
            caps = "yes" if synth.draws_capitals_for_lowercase(str(path)) else ""
            print(
                f"{path.name:<40} {path.parent.name:<8} {caps:<5} {licence[:36]:<36} "
                f"{len(gaps)} missing"
            )
    return 0


def pin() -> int:
    """Print every face's SHA-256 as fetched now, for pinning by hand."""
    session = requests.Session()
    cache = DEST / ".cache"
    for face in [*FACES, *(face for face, _ in VARIABLE)]:
        data = fontsource.download(face.source, cache, session)
        if face.member is not None:
            data = fontsource.member_of(data, face.member)
        print(f"{face.filename}\t{fontsource.sha256_of(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
