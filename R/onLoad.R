DNN_Model <- NULL
ST_RNG <- NULL

.onLoad <- function(libname, pkgname){
  DNN_Model <<- reticulate::import_from_path(
    module = "DNN_Model",
    path = system.file("DNN_Model", package = "DNNSIM"),
    delay_load = TRUE
  )
  ST_RNG <<- reticulate::import_from_path(
    module = "ST_RNG",
    path = system.file("DNN_Model", package = "DNNSIM"),
    delay_load = TRUE
  )
}
