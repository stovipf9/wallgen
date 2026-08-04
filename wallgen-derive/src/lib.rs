use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(PaletteFields)]
pub fn derive_palette_fields(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let struct_name = &ast.ident;

    let named_fields = match ast.data {
        Data::Struct(data_struct) => match data_struct.fields {
            Fields::Named(fields_named) => fields_named.named,
            _ => panic!("PaletteFields only supports struct with named fields"),
        },
        _ => panic!("PaletteFields only supports structs"),
    };

    let field_idents: Vec<_> = named_fields
        .iter()
        .map(|f| f.ident.clone().unwrap())
        .collect();
    let num_field_idents = field_idents.len();

    let field_type = named_fields
        .first()
        .expect("PaletteFields needs at least one field")
        .ty
        .clone();

    if named_fields.iter().any(|f| f.ty != field_type) {
        panic!("all the types of fields must be the same");
    }

    let expanded = quote! {
        impl #struct_name {
            pub fn from_colors(colors: ::std::collections::HashMap<String, #field_type>) -> Result<#struct_name, String> {
                Ok(#struct_name {
                    #( #field_idents: *colors.get(stringify!(#field_idents)).ok_or(format!("missing slot: {}", stringify!(#field_idents)))? ),*
                })
            }
            /// All colors.
            pub fn all(&self) -> [#field_type; #num_field_idents] {
                [
                    #( self.#field_idents ),*
                ]
            }
        }
    };

    expanded.into()
}
