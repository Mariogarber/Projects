install.packages("sylly")
install.packages("sylly.es")
install.packages("stringr")

sacar_ultima_silaba <- function(palabra){
  res <- hyphen(palabra, hyph.pattern="es", min.length = 2, quiet = TRUE)
  silabas <- unlist(strsplit(res@hyphen[["word"]],""))
  numero_silabas = res[[1]]
  guiones_encontrados = 0
  n = 1
  while (guiones_encontrados < numero_silabas - 1){
    if (silabas[n] == '-'){
      guiones_encontrados = guiones_encontrados + 1
    }
    silabas <- silabas[(n+1):length(silabas)]
  }
  ultima_silaba <- chartr("áéíóú", "aeiou",paste(silabas, collapse = ""))
  return (ultima_silaba)
}

sacar_primera_silaba <- function(palabra){
  res <- hyphen(palabra, hyph.pattern="es", min.length = 2, quiet = TRUE)
  silabas <- unlist(strsplit(res@hyphen[["word"]],""))
  if (res[[1]] == 1){
    primera_silaba = res[[2]]
  }
  else{
    guiones_encontrados <- 0
    n <- 1
    primera_silaba = vector()
    while (guiones_encontrados <  1){
      if (silabas[n] == '-'){
        guiones_encontrados = guiones_encontrados + 1
      }
      else{
        primera_silaba[n] <- silabas[n]
      }
      n = n + 1
    }
    primera_silaba <- chartr("áéíóú", "aeiou", paste(primera_silaba, collapse = ""))
  }
  return (primera_silaba)
}

sacar_encadenacion <- function(palabra){
  encadenacion <- c(palabra)
  for(i in 1:10){
    dimension = dim(dic)[1]
    ultima_silaba <- sacar_ultima_silaba(palabra)
    encontrado <- FALSE
    j <- 1
    while (!encontrado & j <= dimension){
      palabra = dic[j,1]
      palabra_letras <- unlist(strsplit(palabra, ""))
      primeras_letras <- paste(palabra_letras[1:str_count(ultima_silaba)],
                               collapse = "")
      if ( primeras_letras == ultima_silaba){
        primera_silaba <- sacar_primera_silaba(palabra)
        if (primera_silaba == ultima_silaba){
          encontrado = TRUE
          encadenacion <- c(encadenacion, palabra)
        }
      }
      print(j)


      j = j + 1
    }
    if (j > dimension){
      print("No se puede continuar la encadenación.")
      break
    }
  }
  return (encadenacion)
}

sacar_encadenacion_ganadora <- function(primera_silaba ,ultima_silaba){
  encadenacion <- c(palabra)
  for(i in 1:10){
    dimension = dim(dic)[1]
    ultima_silaba <- sacar_ultima_silaba(palabra)
    encontrado <- FALSE
    j <- 1
    while (!encontrado & j <= dimension){
      palabra = dic[j,1]
      palabra_letras <- unlist(strsplit(palabra, ""))
      primeras_letras <- paste(palabra_letras[1:str_count(ultima_silaba)],
                               collapse = "")
      if ( primeras_letras == ultima_silaba){
        primera_silaba <- sacar_primera_silaba(palabra)
        if (primera_silaba == ultima_silaba){
          encontrado = TRUE
          encadenacion <- c(encadenacion, palabra)
        }
      }
      print(j)
      j = j + 1
    }
    if (j > dimension){
      print("No se puede continuar la encadenación.")
      break
    }
  }
  return (encadenacion)
}
library(sylly.es)
library(stringr)

dic <- read.table(file='dic_es_clean.txt', encoding='UTF-8')

silabas_ganadoras = list("ñon", "niz", "xo", "drin")
silabas_perdedoras_ñon = list("ca", "ga", "mi", "mo","mu", "pe","pi","ri",
                              "bri", "gra", "gri","gru","gui", "qui")
silabas_perdedoras_niz = list("bar", "co")
silabas_perdedoras_xo = list("fle", "a", "ple", "de","in", "am", "con",
                             "com", "in", "or")
silabas_perdedoras_drin = list("al", "pie", "sa", "ma")

jugando = TRUE
palabra = "disturbio"
encadenacion_global = c(palabra)
ultima_silaba = sacar_ultima_silaba(palabra)

if (length(which(silabas_perdedoras_ñon == ultima_silaba)) > 0){
  encadenacion <- sacar_encadenacion_ganadora(ultima_silaba,
                                              silabas_ganadoras[[1]])
  print("¡El ordenador ha ganado!")
  jugando = FALSE
}else if (length(which(silabas_perdedoras_niz == ultima_silaba)) > 0){
  encadenacion <- sacar_encadenacion_ganadora(ultima_silaba,
                                              silabas_ganadoras[[2]])
  print("¡El ordenador ha ganado!")
  jugando = FALSE
}else if (length(which(silabas_perdedoras_xo == ultima_silaba)) > 0){
  encadenacion <- sacar_encadenacion_ganadora(ultima_silaba,
                                              silabas_ganadoras[[3]])
  print("¡El ordenador ha ganado!")
  jugando = FALSE
}else if (length(which(silabas_perdedoras_drin == ultima_silaba)) > 0){
  encadenacion <- sacar_encadenacion_ganadora(ultima_silaba,
                                              silabas_ganadoras[[4]])
  print("¡El ordenador ha ganado!")
  jugando = FALSE
}else {
  encadenacion <- sacar_encadenacion(palabra)
  if (length(encadenacion) == 1){
    print("¡El usuario ha ganado!")
    jugando = FALSE
  }
  else{
    palabra_actual <- encadenacion[2]
    encadenacion_global <- c(encadenacion_global, encadenacion[2])
  }
}

