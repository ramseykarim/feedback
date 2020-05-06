## some notes from a week ago
## I should write these down more formally and decide what to do with this info

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
April 30, 2020
write this in your notebook tomorrow (May 1)
and write this in that latex document maybe???
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
VPHAS+ (Mohr-Smith 2015) fit for log(Teff), distance modulus mu,
Rv, and A0
They fit only main sequence models with fixed log(g) ~ 4.0 (not sure if == 4.0)
Section 3.2.1 in the paper is a nice description of how they construct their model
They fit log(Teff) between 4.2 (B3V) and 4.7 (available models?)
    this maps to between ~16 and 50 kK

They say that VargasAlvarez2013 fit 29 known OB stars in the center of Wd2
They cross-matched 24 (missing #597, #826, #843, #903 and #906 in VA numbers)
    and chalk it up to resolution
They skip  #896 and #771 for which there is no detection in one or more optical bands
    #771 also is super-blue on their chart (higher than RJ limit, theoretical max)
And they skip #1004 due to incomplete NIR photometry
So they get 21 fittable cross-matches

They show their posteriors, and there's a killer pencil-thin correlation between
    log(Teff) and distance modulus, so should only use their Teff as last resort
    Apparently this is worse with higher Teff and is slightly more constrained
    for later types


Also, we should use the Sternberg tables to map between Teff and log_g (or assume 4.0)
    so that we can translate these Teffs to FUV fluxes and other stuff
Actually, JK, we should try the Vacca calibrations for this somehow.....
    picture this: use the spectral type as a "parameter" and draw out the
    "parametrized" curve of Teff and log_g, and then fit some n-d polynomial
    to one as a function of the other so we can map between them. This should
    let us use PoWR as main-sequence only
The Sternberg class V table ranges from log(Teff) 4.506 to 4.710, which is honestly
    probably the entire uncertainty range we'll get from the VPHAS+ Teffs
Plus we can use the lower/upper limits they give as our uncertainties
I should make a class with some of the same functions as STResolver that can
    start from Teff and move to flux and stuff, and expect Teff errors rather
    than spectral type possibilities (which is what STResolver is built for)
With PoWR at log_g == 4, we get a range of Teff between ~19 kK and 49 kK
    These limits get to weird edges (too high log_g for the cool end and too low
    gravity for the hot end), so we probably want to use that Vacca idea to
    correct log_g a bit more
This will get a liiiiitle complicated but hopefully we can keep it short.

I should find out how many of those matched TFT rows had VPHAS known spectral types
I should figure out which stars VA found
I need to revisit / straighten out all this stuff and WRITE IT DOWN IN THE LATEX
DOCUMENT and also write all this down there too cause this is good stuff
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

May 2, 2020:
I also need to figure out what VPHAS+ considered early type; I think it was
a Teff cut? JK it was first a color selection (bluer than B something) then the
fits and then maybe(?) a Teff cut along with an extinction criteria based on
known(?) Wd2 stars
It looks like they put "WD2" in the Notes column if they suspected the OB candidate
to be associated with the cluster
