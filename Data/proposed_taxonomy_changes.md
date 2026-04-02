# Propuestas de cambio en `unified_df.csv`
Este archivo documenta los cambios propuestos para registros de `Data/unified_df.csv` con problemas taxonómicos detectados localmente.

> Auditoría realizada contra `Data/unified_df.csv` en este repositorio. No se ha modificado el CSV aún; esto documenta cambios propuestos para tu aprobación.

## Resumen de auditoría
- Total de filas en `unified_df.csv`: 225
- Filas con problemas taxonómicos estructurales detectados: 48
- Problemas más frecuentes: `Microchiroptera` como suborden antiguo; duplicación `Family == Sub-Family`; suborden marsupial mal formado; orden/superorden duplicados; faltas de `Sub-Order` en órdenes donde no hay suborden estándar; errores tipográficos de nombre científico o de género; el caso aislado de `Ocotdon degus`.

## Correcciones propuestas por fila
### Id 3 — `Armadillo`
- Actual: `Species = Chaetophractus villosus`; `Genus = Chaetophractus`; `Sub-Family = Euphractinae`; `Family = Chlamyphoridae`; `Sub-Order = Cingulata`; `Order = Xenarthra`; `Super-Order = Xenarthra`; `Common-Name = Big hairy armadillo`
- Propuesto: Set `Order = Cingulata`, clear `Sub-Order` (or leave blank), keep `Super-Order = Xenarthra`.

### Id 4 — `ArtibeusJamacien`
- Actual: `Species = Artibeus jamaicensis`; `Genus = Artibeus`; `Sub-Family = Stenodermatinae`; `Family = Phyllostomidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Jamaican fruit bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 5 — `Asellia`
- Actual: `Species = Asellia tridens`; `Genus = Asellia`; `Sub-Family = Hipposideridae`; `Family = Hipposideridae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Trident leaf-nosed bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`; correct `Sub-Family`/`Family` relationship (e.g. `Sub-Family = Hipposiderinae`, `Family = Hipposideridae`).

### Id 14 — `Battongia1`
- Actual: `Species = Bettongia setosa`; `Genus = Bettongia`; `Sub-Family = Potoroinae`; `Family = Potoroidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Rat Kangaroo (Rufous bettong)`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 15 — `Battongia2`
- Actual: `Species = Bettongia setosa`; `Genus = Bettongia`; `Sub-Family = Potoroinae`; `Family = Potoroidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Rat Kangaroo (Rufous bettong)`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 19 — `BottleNoseDolphin`
- Actual: `Species = Tursiops truncatus`; `Genus = Tursiops`; `Sub-Family = Delphinidae`; `Family = Delphinidae`; `Sub-Order = Odontoceti`; `Order = Cetartiodactyla`; `Super-Order = Laurasiatheria`; `Common-Name = Bottlenose dolphin`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or a true subfamily such as `Delphininae`.

### Id 32 — `CarolliaPerspicilatta1`
- Actual: `Species = Carollia perspicilatta`; `Genus = carollia`; `Sub-Family = Carolliinae`; `Family = Phyllostomidae`; `Sub-Order = Yangochiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Seba's short-tailed bat`
- Propuesto: Capitalise genus: `Genus = Carollia`.

### Id 33 — `CarolliaPerspicilatta2`
- Actual: `Species = Carollia perspicilatta`; `Genus = carollia`; `Sub-Family = Carolliinae`; `Family = Phyllostomidae`; `Sub-Order = Yangochiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Seba's short-tailed bat`
- Propuesto: Capitalise genus: `Genus = Carollia`.

### Id 41 — `ChaerephonPlicata`
- Actual: `Species = Chaerephon plicata`; `Genus = Chaerephon`; `Sub-Family = Molossinae`; `Family = Molossidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Wrinkle-lipped free tailed bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 51 — `CommonTreeShrew`
- Actual: `Species = Tupaia glis`; `Genus = Tupaia`; `Sub-Family = `; `Family = Tupaiidae`; `Sub-Order = `; `Order = Scandentia`; `Super-Order = Euarchontoglires`; `Common-Name = Common treeshrew`
- Propuesto: No standard suborder for Scandentia; leave `Sub-Order` blank or use a consistent placeholder rather than an empty string if needed.

### Id 57 — `Dego`
- Actual: `Species = Ocotdon degus`; `Genus = Octodon`; `Sub-Family = `; `Family = Octodontidae`; `Sub-Order = Hystricomorpha`; `Order = Rodentia`; `Super-Order = Euarchontoglires`; `Common-Name = Common degu`
- Propuesto: Fix species spelling: `Species = Octodon degus`.

### Id 69 — `Eptesicus1`
- Actual: `Species = Eptesicus fuscus`; `Genus = Eptesicus`; `Sub-Family = Vespertilioninae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Big brown bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 70 — `Eptesicus2`
- Actual: `Species = Eptesicus fuscus`; `Genus = Eptesicus`; `Sub-Family = Vespertilioninae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Big brown bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 93 — `GrayKangaroo1`
- Actual: `Species = Macropus giganteus`; `Genus = Macropus`; `Sub-Family = Macropodinae`; `Family = Macropodidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Eastern grey kangaroo`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 94 — `GrayKangaroo3`
- Actual: `Species = Macropus giganteus`; `Genus = Macropus`; `Sub-Family = Macropodinae`; `Family = Macropodidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Eastern grey kangaroo`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 97 — `Hare`
- Actual: `Species = Lepus europaeus`; `Genus = Lepus`; `Sub-Family = Leporinae`; `Family = Leporidae`; `Sub-Order = `; `Order = Lagomorpha`; `Super-Order = Euarchontoglires`; `Common-Name = Hare`
- Propuesto: No standard suborder for Lagomorpha; leave `Sub-Order` blank or use a consistent placeholder.

### Id 98 — `HimalianBear`
- Actual: `Species = Ursus arctos isabellinus`; `Genus = Ursus`; `Sub-Family = Ursidae`; `Family = Ursidae`; `Sub-Order = Caniformia`; `Order = Carnivora`; `Super-Order = Laurasiatheria`; `Common-Name = Himalayan black bear`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or a valid subfamily such as `Ursinae`.

### Id 99 — `HippArmiger`
- Actual: `Species = Hipposideros armiger`; `Genus = Hipposideros`; `Sub-Family = Hipposideridae`; `Family = Hipposideridae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Great roundleaf bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`; fix duplicate family/subfamily by using `Sub-Family = Hipposiderinae` or blank.

### Id 103 — `HowlerMonkey1`
- Actual: `Species = `; `Genus = Aloutta`; `Sub-Family = Alouattinae`; `Family = Atelidae`; `Sub-Order = Haplorrini`; `Order = Primates`; `Super-Order = Euarchontoglires`; `Common-Name = Black and gold howler`
- Propuesto: Set `Genus = Alouatta`; fill `Species = Alouatta seniculus`; set `Sub-Order = Haplorrhini`.

### Id 104 — `HowlerMonkey2`
- Actual: `Species = `; `Genus = Aloutta`; `Sub-Family = Alouattinae`; `Family = Atelidae`; `Sub-Order = Haplorrini`; `Order = Primates`; `Super-Order = Euarchontoglires`; `Common-Name = Black and gold howler`
- Propuesto: Set `Genus = Alouatta`; fill `Species = Alouatta seniculus`; set `Sub-Order = Haplorrhini`.

### Id 107 — `Hyrax1`
- Actual: `Species = Procavia capensis`; `Genus = Procavia`; `Sub-Family = `; `Family = Procaviidae`; `Sub-Order = `; `Order = Hyracoidea`; `Super-Order = Afrotheria`; `Common-Name = Rock hyrax`
- Propuesto: No standard suborder for Hyracoidea; leave `Sub-Order` blank or use a consistent placeholder.

### Id 108 — `Hyrax2`
- Actual: `Species = Procavia capensis`; `Genus = Procavia`; `Sub-Family = `; `Family = Procaviidae`; `Sub-Order = `; `Order = Hyracoidea`; `Super-Order = Afrotheria`; `Common-Name = Rock hyrax`
- Propuesto: No standard suborder for Hyracoidea; leave `Sub-Order` blank or use a consistent placeholder.

### Id 111 — `Koala`
- Actual: `Species = Phascolarctos cinereus`; `Genus = Phascolarctos`; `Sub-Family = `; `Family = Phascolarctidae`; `Sub-Order = `; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Koala`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Vombatiformes`.

### Id 143 — `MiniopterusSchreibersii1`
- Actual: `Species = Miniopterus schreibersii`; `Genus = Miniopterus`; `Sub-Family = `; `Family = Miniopteridae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Common bent wing bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 146 — `MyotisEmargenitus`
- Actual: `Species = Myotis emarginatus`; `Genus = Myotis`; `Sub-Family = Myotinae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Geoffroy's bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 147 — `MyotisMyotis`
- Actual: `Species = Myotis myotis`; `Genus = Myotis`; `Sub-Family = Myotinae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Greater mouse-eared bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 148 — `MyotisVivesi1`
- Actual: `Species = Myotis Vivesi`; `Genus = Myotis`; `Sub-Family = Myotinae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Fish eating bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 163 — `Pecari1`
- Actual: `Species = Pecari`; `Genus = Pecari`; `Sub-Family = Tayassuidae`; `Family = Tayassuidae`; `Sub-Order = Suina`; `Order = Cetartiodactyla`; `Super-Order = Laurasiatheria`; `Common-Name = Pecari`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or a valid subfamily for Pecari.

### Id 164 — `Pecari2`
- Actual: `Species = Pecari`; `Genus = Pecari`; `Sub-Family = Tayassuidae`; `Family = Tayassuidae`; `Sub-Order = Suina`; `Order = Cetartiodactyla`; `Super-Order = Laurasiatheria`; `Common-Name = Pecari`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or a valid subfamily for Pecari.

### Id 165 — `Pkuhlii2`
- Actual: `Species = Pipistrellus kuhlii`; `Genus = Pipistrellus`; `Sub-Family = Vespertilioninae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Kuhl's pipistrelle`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 166 — `Pkuhlii3`
- Actual: `Species = Pipistrellus kuhlii`; `Genus = Pipistrellus`; `Sub-Family = Vespertilioninae`; `Family = Vespertilionidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Kuhl's pipistrelle`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 173 — `Rabbit`
- Actual: `Species = Oryctolagus cuniculus`; `Genus = Oryctolagus`; `Sub-Family = Leporidae`; `Family = Leporidae`; `Sub-Order = `; `Order = Lagomorpha`; `Super-Order = Euarchontoglires`; `Common-Name = Rabbit`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family = Leporinae`, keep `Family = Leporidae`.

### Id 180 — `RedKangaroo1`
- Actual: `Species = Macropus rufus`; `Genus = Macropus`; `Sub-Family = Macropodinae`; `Family = Macropodidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Red Kangaroo`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 181 — `RedKangaroo2`
- Actual: `Species = Macropus rufus`; `Genus = Macropus`; `Sub-Family = Macropodinae`; `Family = Macropodidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Red Kangaroo`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 182 — `RedKangaroo3`
- Actual: `Species = Macropus rufus`; `Genus = Macropus`; `Sub-Family = Macropodinae`; `Family = Macropodidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Red Kangaroo`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 183 — `RedNeckWallaby`
- Actual: `Species = Macropus rufogriseus`; `Genus = Macropus`; `Sub-Family = Macropodinae`; `Family = Macropodidae`; `Sub-Order = MacropodiformesPotoroidae`; `Order = Marsupialia`; `Super-Order = No placenta`; `Common-Name = Red neck wallaby`
- Propuesto: Set `Order = Diprotodontia`, `Super-Order = Marsupialia`, `Sub-Order = Macropodiformes`.

### Id 185 — `Rhinolophus`
- Actual: `Species = Rhinolophus ferrumequinum`; `Genus = Rhinolophus`; `Sub-Family = Rhinolophinae`; `Family = Rhinolophidae`; `Sub-Order = Yinpterochiropteria`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Greater horseshow bat`
- Propuesto: Set `Sub-Order = Yinpterochiroptera`; fix common name to `Greater horseshoe bat`.

### Id 186 — `Rhinopoma`
- Actual: `Species = Rhinopoma hardwickii`; `Genus = Rhinopoma`; `Sub-Family = Rhinopomatidae`; `Family = Rhinopomatidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Greater mouse-tailed bat`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or a valid Rhinopomatidae subfamily; set `Sub-Order = Yangochiroptera`.

### Id 195 — `StrippedDolphin1`
- Actual: `Species = Stenella coeruleoalba`; `Genus = Stenella`; `Sub-Family = Delphinidae`; `Family = Delphinidae`; `Sub-Order = Odontoceti`; `Order = Cetartiodactyla`; `Super-Order = Laurasiatheria`; `Common-Name = Stripped dolphin`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or `Delphininae`.

### Id 196 — `StrippedDolphin2`
- Actual: `Species = Stenella coeruleoalba`; `Genus = Stenella`; `Sub-Family = Delphinidae`; `Family = Delphinidae`; `Sub-Order = Odontoceti`; `Order = Cetartiodactyla`; `Super-Order = Laurasiatheria`; `Common-Name = Stripped dolphin`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family` should be blank or `Delphininae`.

### Id 197 — `SturniraLilium`
- Actual: `Species = Sturnira lilium`; `Genus = Sturnira`; `Sub-Family = Stenodermatinae`; `Family = Phyllostomidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Little yellow-shouldered bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 198 — `TadaridaTeniotis1`
- Actual: `Species = Tadarida teniotis`; `Genus = Tadarida`; `Sub-Family = Molossinae`; `Family = Molossidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = European free-tailed bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 199 — `TadaridaTeniotis2`
- Actual: `Species = Tadarida teniotis`; `Genus = Tadarida`; `Sub-Family = Molossinae`; `Family = Molossidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = European free-tailed bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 209 — `VampireBat1`
- Actual: `Species = Desmodus rotundus`; `Genus = Desmodus`; `Sub-Family = Desmodontinae`; `Family = Phyllostomidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Common vampire bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 210 — `VampireBat2`
- Actual: `Species = Desmodus rotundus`; `Genus = Desmodus`; `Sub-Family = Desmodontinae`; `Family = Phyllostomidae`; `Sub-Order = Microchiroptera`; `Order = Chiroptera`; `Super-Order = Laurasiatheria`; `Common-Name = Common vampire bat`
- Propuesto: Set `Sub-Order = Yangochiroptera`.

### Id 216 — `WildRabbit1`
- Actual: `Species = Oryctolagus cuniculus`; `Genus = Oryctolagus`; `Sub-Family = Leporidae`; `Family = Leporidae`; `Sub-Order = `; `Order = Lagomorpha`; `Super-Order = Euarchontoglires`; `Common-Name = European Rabbit`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family = Leporinae`, keep `Family = Leporidae`.

### Id 217 — `WildRabbit2`
- Actual: `Species = Oryctolagus cuniculus`; `Genus = Oryctolagus`; `Sub-Family = Leporidae`; `Family = Leporidae`; `Sub-Order = `; `Order = Lagomorpha`; `Super-Order = Euarchontoglires`; `Common-Name = European Rabbit`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family = Leporinae`, keep `Family = Leporidae`.

### Id 218 — `WildRabbit3`
- Actual: `Species = Oryctolagus cuniculus`; `Genus = Oryctolagus`; `Sub-Family = Leporidae`; `Family = Leporidae`; `Sub-Order = `; `Order = Lagomorpha`; `Super-Order = Euarchontoglires`; `Common-Name = European Rabbit`
- Propuesto: Fix duplicate family/subfamily: `Sub-Family = Leporinae`, keep `Family = Leporidae`.

## Notas generales
- El dataset usa una mezcla de esquemas taxonómicos antiguos y modernos, especialmente en Chiroptera (`Microchiroptera` vs `Yangochiroptera` / `Yinpterochiroptera`).
- Para las filas con `Family == Sub-Family`, mi recomendación es no mantener duplicados: dejar `Sub-Family` vacío si no existe una subfamilia específica o usar la subfamilia válida cuando se conozca. Ejemplos claros: `Tursiops`, `Stenella` y `Oryctolagus cuniculus`.
- Los marsupiales listados con `Order = Marsupialia` y `Super-Order = No placenta` deben actualizarse a `Order = Diprotodontia`, `Super-Order = Marsupialia` y, cuando corresponda, `Sub-Order = Macropodiformes` o `Vombatiformes` para el koala.
- Si quieres, puedo aplicar estos cambios directamente al CSV en un segundo paso y dejar un backup del archivo original.